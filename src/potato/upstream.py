"""HTTP client that forwards OpenAI-compatible requests to NVIDIA NIM."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from potato.balancer import KeyPool, KeyStats
from potato.safety.backoff import sleep_backoff

logger = logging.getLogger(__name__)


def parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After header to seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


class UpstreamClient:
    def __init__(
        self,
        base_url: str,
        pool: KeyPool,
        timeout: float = 300.0,
        *,
        connect_timeout: float = 5.0,
        user_agent: str | None = None,
        proxy_url: str | None = None,
        retry_backoff_base: float = 0.5,
        retry_backoff_cap: float = 16.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.pool = pool
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.user_agent = user_agent
        self.proxy_url = proxy_url
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_cap = retry_backoff_cap
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": httpx.Timeout(self.timeout, connect=self.connect_timeout),
            "follow_redirects": True,
            "limits": httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
                keepalive_expiry=30.0,
            ),
        }
        if self.proxy_url:
            kwargs["proxy"] = self.proxy_url
            logger.info("upstream using egress proxy %s", self.proxy_url.split("@")[-1])
        self._client = httpx.AsyncClient(**kwargs)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("UpstreamClient not started")
        return self._client

    def _headers(
        self,
        key: KeyStats,
        extra: dict[str, str] | None = None,
        *,
        accept_stream: bool = False,
    ) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {key.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if accept_stream else "application/json",
        }
        if self.user_agent:
            h["User-Agent"] = self.user_agent
        if extra:
            for k, v in extra.items():
                if k.lower() in {"authorization", "host", "content-length"}:
                    continue
                h[k] = v
        return h

    @staticmethod
    def _filter_headers(headers: httpx.Headers, *, streaming: bool = False) -> dict[str, str]:
        # For streaming responses, preserve connection-related headers so
        # downstream clients (Cursor agent mode, etc.) keep the SSE socket open.
        skip = {
            "content-encoding",
            "transfer-encoding",
            "content-length",
        }
        if not streaming:
            skip |= {"connection", "keep-alive"}
        return {k: v for k, v in headers.items() if k.lower() not in skip}

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        forward_headers: dict[str, str] | None = None,
        max_retries: int = 3,
        preferred_key_id: str | None = None,
    ) -> tuple[int, Any, dict[str, str], KeyStats]:
        """
        Non-streaming request with key rotation + retry on 429.
        Returns (status_code, body_json_or_text, response_headers, key).
        """
        last_error: BaseException | None = None
        for attempt in range(max_retries):
            key = await self.pool.acquire(preferred_key_id=preferred_key_id)
            released = False
            started = time.monotonic()
            try:
                resp = await self.client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    headers=self._headers(key, forward_headers),
                )
                latency = time.monotonic() - started
                rate_limited = resp.status_code == 429
                success = 200 <= resp.status_code < 300
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                await self.pool.release(
                    key,
                    success=success,
                    latency=latency if success else None,
                    rate_limited=rate_limited,
                    status_code=resp.status_code,
                    retry_after_seconds=retry_after,
                )
                released = True

                if rate_limited and attempt < max_retries - 1:
                    delay = await sleep_backoff(
                        attempt,
                        base=self.retry_backoff_base,
                        cap=self.retry_backoff_cap,
                        retry_after=retry_after,
                    )
                    logger.info(
                        "upstream 429 on %s; backoff %.2fs then rotate key (attempt %s)",
                        key.key_id,
                        delay,
                        attempt + 1,
                    )
                    continue

                content_type = resp.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        body = resp.json()
                    except Exception as exc:
                        # Malformed upstream JSON is a transport-class failure:
                        # raise so the fallback executor advances the chain.
                        raise RuntimeError(
                            f"upstream returned invalid JSON body (HTTP {resp.status_code})"
                        ) from exc
                else:
                    body = resp.text

                # Auth failures: rotate key and retry (stale/revoked key)
                if (
                    resp.status_code in {401, 403}
                    and attempt < max_retries - 1
                    and len(self.pool) > 1
                ):
                    logger.info(
                        "upstream HTTP %s on %s; rotating key (attempt %s)",
                        resp.status_code,
                        key.key_id,
                        attempt + 1,
                    )
                    continue

                # 5xx: return immediately — fallback executor handles model-level
                # failover. Retrying the same provider wastes deadline seconds.
                return resp.status_code, body, self._filter_headers(resp.headers), key
            except BaseException as exc:
                # Cancel-safe: release key even on cancellation/task abort
                if not released:
                    await self.pool.release(key, success=False, latency=None)
                if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                    raise
                last_error = exc
                logger.exception("upstream error on %s: %s", key.key_id, exc)
                if attempt >= max_retries - 1:
                    raise
                delay = await sleep_backoff(
                    attempt,
                    base=self.retry_backoff_base,
                    cap=self.retry_backoff_cap,
                )
                logger.info(
                    "upstream transport error; backoff %.2fs (attempt %s)",
                    delay,
                    attempt + 1,
                )
        raise RuntimeError(f"upstream failed after retries: {last_error}")

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        forward_headers: dict[str, str] | None = None,
        max_retries: int = 3,
        preferred_key_id: str | None = None,
    ) -> tuple[int, AsyncIterator[bytes], dict[str, str], KeyStats]:
        """
        Open a streaming response. Caller must consume the iterator fully
        so the key is released (wrapper handles release on completion/error).
        """
        last_error: BaseException | None = None
        for attempt in range(max_retries):
            key = await self.pool.acquire(preferred_key_id=preferred_key_id)
            released = False
            started = time.monotonic()
            try:
                req = self.client.build_request(
                    method,
                    path,
                    json=json_body,
                    headers=self._headers(key, forward_headers, accept_stream=True),
                )
                resp = await self.client.send(req, stream=True)

                if resp.status_code == 429:
                    retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                    ra_raw = resp.headers.get("Retry-After")
                    await resp.aclose()
                    await self.pool.release(
                        key,
                        success=False,
                        rate_limited=True,
                        status_code=429,
                        retry_after_seconds=retry_after,
                    )
                    released = True
                    if attempt < max_retries - 1:
                        delay = await sleep_backoff(
                            attempt,
                            base=self.retry_backoff_base,
                            cap=self.retry_backoff_cap,
                            retry_after=retry_after,
                        )
                        logger.info(
                            "upstream stream 429 on %s; backoff %.2fs then rotate (attempt %s)",
                            key.key_id,
                            delay,
                            attempt + 1,
                        )
                        continue

                    from potato.compat import frame_sse_error

                    payload = frame_sse_error(
                        "Rate limited by upstream",
                        code="rate_limit_exceeded",
                        status=429,
                        retry_after=str(ra_raw) if ra_raw else None,
                    )

                    async def err_429(p: bytes = payload) -> AsyncIterator[bytes]:
                        yield p

                    headers_out: dict[str, str] = {"content-type": "text/event-stream"}
                    if ra_raw:
                        headers_out["Retry-After"] = str(ra_raw)
                    return (429, err_429(), headers_out, key)

                if resp.status_code in {401, 403}:
                    await resp.aclose()
                    await self.pool.release(key, success=False, status_code=resp.status_code)
                    released = True
                    if attempt < max_retries - 1:
                        continue

                    from potato.compat import frame_sse_error

                    payload = frame_sse_error(
                        f"Upstream authentication failed (HTTP {resp.status_code})",
                        code="upstream_auth_error",
                        status=resp.status_code,
                    )

                    async def err_auth(
                        p: bytes = payload, st: int = resp.status_code
                    ) -> AsyncIterator[bytes]:
                        yield p

                    return (
                        resp.status_code,
                        err_auth(),
                        {"content-type": "text/event-stream"},
                        key,
                    )

                # 5xx: return immediately — fallback executor handles model failover
                out_headers = self._filter_headers(resp.headers, streaming=True)
                status_code = resp.status_code
                bound_key = key
                bound_resp = resp
                bound_started = started
                bound_status = status_code

                async def byte_iter(
                    key: KeyStats = bound_key,
                    resp: httpx.Response = bound_resp,
                    started: float = bound_started,
                    status: int = bound_status,
                ) -> AsyncIterator[bytes]:
                    success = 200 <= status < 300
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    except BaseException:
                        # Cancel-safe: mark failure on cancellation/generator close
                        success = False
                        raise
                    finally:
                        latency = time.monotonic() - started
                        await resp.aclose()
                        await self.pool.release(
                            key,
                            success=success,
                            latency=latency if success else None,
                            rate_limited=status == 429,
                            status_code=status,
                        )

                released = True  # key release is now the byte_iter's responsibility
                return status_code, byte_iter(), out_headers, key
            except BaseException as exc:
                # Cancel-safe: release key even on cancellation/task abort
                if not released:
                    await self.pool.release(key, success=False)
                if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                    raise
                last_error = exc
                logger.exception("upstream stream error on %s: %s", key.key_id, exc)
                if attempt >= max_retries - 1:
                    raise
                delay = await sleep_backoff(
                    attempt,
                    base=self.retry_backoff_base,
                    cap=self.retry_backoff_cap,
                )
                logger.info(
                    "upstream stream transport error; backoff %.2fs (attempt %s)",
                    delay,
                    attempt + 1,
                )
        raise RuntimeError(f"upstream stream failed after retries: {last_error}")
