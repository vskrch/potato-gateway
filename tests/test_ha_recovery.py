"""HA remediation regressions (P0/P1): T1–T10 + RCA promotion.

Each test pins one defect from docs/ha-audit-report.md so the 503
"Upstream Request Failed" failure modes cannot return.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from potato.balancer import KeyPool, KeyStats
from potato.catalog import ModelRegistry
from potato.config import Settings
from potato.routing import FallbackExecutor, Intent, RouteDecision
from potato.safety.circuit_breaker import BreakerState, ProviderCircuitBreaker

YAML = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def _key(i: int = 0) -> KeyStats:
    return KeyStats(key_id=f"key-{i}", api_key=f"k{i}")


def _auto(chain: list[str], requested: str = "potato/auto") -> RouteDecision:
    return RouteDecision(
        chain=chain,
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model=requested,
    )


def _hub_stub(cb: ProviderCircuitBreaker, clients: dict[str, object]) -> SimpleNamespace:
    """Minimal hub: provider_ids + circuit_breaker + client_for_model."""
    from potato.catalog.hub import ProviderCircuitOpen

    def client_for_model(model_id: str) -> tuple[object, str, str]:
        pid = model_id.split("/")[0] if "/" in model_id else "nim"
        if not cb.allow(pid):
            raise ProviderCircuitOpen(f"provider '{pid}' circuit is open")
        if model_id not in clients:
            raise RuntimeError(f"provider '{pid}' is not available for model '{model_id}'")
        return clients[model_id], pid, model_id

    return SimpleNamespace(
        provider_ids={"nim", "p1", "p2"},
        circuit_breaker=cb,
        client_for_model=client_for_model,
        has_runtime=lambda pid: True,
    )


# ── T1: recovery runs when the primary chain consumes the whole budget (D1) ──


@pytest.mark.asyncio
async def test_recovery_runs_when_chain_sized_to_budget() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        intent_max_fallbacks={"chat_fast": 6},
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    live = {f"model-{i}" for i in range(7)}
    reg.live_ids = live
    calls: list[str] = []
    rescue: str = ""

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        calls.append(model)
        if model == rescue:
            return 200, {"choices": [{"message": {"content": "ok"}}]}, {}, _key(1)
        return 503, {"error": {"message": "Upstream Request Failed"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    chain = ex._chain(_auto(sorted(live)[:6]), had_tools=False)
    assert len(chain) == 6  # primary chain consumes the whole chain budget
    # The rescue model must sit OUTSIDE the built primary chain: only the
    # recovery phases can reach it.
    rescue = next(m for m in sorted(live) if m not in chain)
    # Pin the primary chain: execute_json rebuilds via _chain(), which would
    # otherwise re-expand from the live pool and swallow the rescue model.
    ex._chain = lambda d, had_tools=False: list(chain)  # type: ignore[method-assign]

    result = await ex.execute_json("/chat/completions", {"messages": []}, _auto(chain))
    assert result.status_code == 200
    assert result.model == rescue
    assert len(calls) > len(chain)  # recovery phases actually ran


# ── T2: circuit-open skip never trips the breaker (D2) ──


@pytest.mark.asyncio
async def test_stream_circuit_skip_does_not_trip_breaker() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"p1/a", "p1/b"}
    cb = ProviderCircuitBreaker(failure_threshold=6, recovery_timeout=60.0)
    cb.force_allow("p1")  # HALF_OPEN: exactly one probe slot
    assert cb.state("p1") == BreakerState.HALF_OPEN

    ex = FallbackExecutor(AsyncMock(), reg, settings)
    ex.hub = _hub_stub(cb, {})  # type: ignore[assignment]
    ex._provider_id_for = lambda m: "p1"  # type: ignore[method-assign]
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]

    decision = _auto(["p1/a", "p1/b"])
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 503
    # Skips must not re-open the half-open breaker.
    assert cb.state("p1") == BreakerState.HALF_OPEN


# ── T3: model-tier errors never open the provider breaker (D3) ──


@pytest.mark.asyncio
async def test_model_errors_do_not_open_provider_breaker() -> None:
    import time as _time

    settings = Settings(
        nim_api_keys=["k"],
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c", "model-d"}

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        mapping = {
            "model-a": (404, {"error": {"message": "model not found"}}),
            "model-b": (429, {"error": {"message": "rate limited"}}),
            "model-c": (400, {"error": {"message": "tools unsupported"}}),
            "model-d": (
                200,
                {"choices": [{"message": {"content": ""}}]},  # empty reply soft-fail
            ),
        }
        status, body = mapping[model]
        return status, body, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    cb = ProviderCircuitBreaker(failure_threshold=6)
    ex = FallbackExecutor(upstream, reg, settings, hub=_hub_stub(cb, {}))
    ex._client_for = lambda m: (upstream, m)  # type: ignore[method-assign]
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._provider_id_for = lambda m: "nim"  # type: ignore[method-assign]

    decision = _auto(["model-a", "model-b", "model-c", "model-d"])
    last = await ex._try_models(
        ["model-a", "model-b", "model-c", "model-d"],
        body={"messages": []},
        path="/chat/completions",
        decision=decision,
        deadline=_time.monotonic() + 60.0,
        forward_headers=None,
        preferred_key_id=None,
        had_tools=False,
        chain_len=4,
        start_idx=0,
        last=None,  # type: ignore[arg-type]
        max_attempts=10,
    )
    assert last is not None
    assert cb.state("nim") == BreakerState.CLOSED
    for m in ("model-a", "model-b", "model-c", "model-d"):
        assert cb.is_model_on_cooldown(m), m


# ── T4: last-resort recovery is scoped (D4) ──


@pytest.mark.asyncio
async def test_last_resort_is_scoped() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_json(method, path, **kwargs):
        return 503, {"error": {"message": "down"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    cb = ProviderCircuitBreaker(failure_threshold=6)
    for _ in range(6):
        cb.fail("p2", is_transport=True)
    assert cb.state("p2") == BreakerState.OPEN
    # Unrelated model cooling down — recovery must not touch it.
    reg.health.record_outcome("unrelated-model", success=False, status_code=429)
    assert reg.health.is_unhealthy("unrelated-model")

    ex = FallbackExecutor(upstream, reg, settings, hub=_hub_stub(cb, {"model-a": upstream}))
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._provider_id_for = lambda m: "p1"  # type: ignore[method-assign]

    result = await ex.execute_json("/chat/completions", {"messages": []}, _auto(["model-a"]))
    assert result.status_code == 503
    assert reg.health.is_unhealthy("unrelated-model")
    assert cb.state("p2") == BreakerState.OPEN


# ── T5: Retry-After sleep is bounded by the deadline (D5b) ──


@pytest.mark.asyncio
async def test_retry_after_sleep_bounded_by_deadline() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_json(method, path, **kwargs):
        return (
            429,
            {"error": {"message": "rate limited"}},
            {"Retry-After": "3600"},
            _key(),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]

    t0 = time.monotonic()
    result = await ex.execute_json("/chat/completions", {"messages": []}, _auto(["model-a", "model-b"]))
    elapsed = time.monotonic() - t0
    assert result.status_code == 503
    assert elapsed < 10.0


# ── T6: per-user rate limit returns 429, not 500 (D5c) ──


def test_user_rate_limit_returns_429() -> None:
    from fastapi.testclient import TestClient

    from potato.main import create_app

    app = create_app(
        Settings(
            allow_insecure_auth=True,
            routing_enabled=False,
            nim_api_keys=["k"],
            proxy_api_keys=["pk"],
            user_rpm_limit=1,
        )
    )

    async def fake_json(method, path, **kwargs):
        return 200, {"choices": [{"message": {"content": "ok"}}]}, {}, _key()

    with TestClient(app) as client:
        client.app.state.upstream.request_json = fake_json  # type: ignore[method-assign]
        headers = {"Authorization": "Bearer pk"}
        body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
        first = client.post("/v1/chat/completions", json=body, headers=headers)
        assert first.status_code == 200
        second = client.post("/v1/chat/completions", json=body, headers=headers)
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "user_rate_limited"


# ── T7: fresh phase draws from the any-live pool (D5a) ──


@pytest.mark.asyncio
async def test_fresh_phase_draws_from_any_live_pool() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        if model == "model-b":
            return 200, {"choices": [{"message": {"content": "b"}}]}, {}, _key(1)
        return 503, {"error": {"message": "down"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    # Deterministic chain builder: only model-a. Recovery must still find
    # model-b via the widest-net pool.
    ex._chain = lambda d, had_tools=False: ["model-a"]  # type: ignore[method-assign]

    result = await ex.execute_json("/chat/completions", {"messages": []}, _auto(["model-a"]))
    assert result.status_code == 200
    assert result.model == "model-b"


# ── T8: terminal envelope carries Retry-After, no raw provider text (D5d) ──


@pytest.mark.asyncio
async def test_terminal_envelope_has_retry_after_and_no_raw_last_body() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=2,
        intent_max_fallbacks={"chat_fast": 2},
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_json(method, path, **kwargs):
        return 503, {"error": {"message": "Upstream Request Failed"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]

    result = await ex.execute_json("/chat/completions", {"messages": []}, _auto(["model-a"]))
    assert result.status_code == 503
    assert result.body["error"]["code"] == "all_providers_failed"
    assert result.headers.get("Retry-After") == "3"
    assert "Upstream Request Failed" not in result.body["error"]["message"]


# ── T9: mid-stream provider error frame marks failure (D6a) ──


@pytest.mark.asyncio
async def test_midstream_provider_error_frame_marks_failure() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=3,
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_stream(method, path, **kwargs):
        async def it():
            yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield b'data: {"error": {"message": "provider blew up", "type": "server_error"}}\n\n'
            yield b"data: [DONE]\n\n"

        return 200, it(), {"content-type": "text/event-stream"}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._client_for = lambda m: (upstream, m)  # type: ignore[method-assign]

    result = await ex.execute_stream("/chat/completions", {"messages": []}, _auto(["model-a"]))
    assert result.status_code == 200  # stream opened; failure surfaces mid-stream
    body = b"".join([chunk async for chunk in result.byte_iter])
    assert result.stream_failed is True
    assert b"upstream_stream_error" in body
    h = reg.health._by_model.get("model-a")
    assert h is not None and h.error_count >= 1


# ── T10: abandoning the stream releases the key without GC (D6b) ──


@pytest.mark.asyncio
async def test_generator_close_releases_key_inflight() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=3,
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}
    pool = KeyPool(api_keys=["k"], max_in_flight_per_key=1)
    key = await pool.acquire()

    class HangIter:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> HangIter:
            return self

        async def __anext__(self) -> bytes:
            import asyncio as _aio

            if self.closed:
                raise StopAsyncIteration
            await _aio.sleep(0.01)
            return b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n'

        async def aclose(self) -> None:
            self.closed = True
            await pool.release(key, success=False)

    async def fake_stream(method, path, **kwargs):
        return 200, HangIter(), {"content-type": "text/event-stream"}, key

    upstream = AsyncMock()
    upstream.stream = fake_stream
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._client_for = lambda m: (upstream, m)  # type: ignore[method-assign]

    result = await ex.execute_stream(
        "/chat/completions", {"messages": [], "stream": True}, _auto(["model-a"])
    )
    assert result.status_code == 200
    it = result.byte_iter
    first = await it.__anext__()
    assert first
    await it.aclose()
    assert key.in_flight == 0


# ── T11: parallel hedge serves the fast secondary on primary stall (P2-1) ──


@pytest.mark.asyncio
async def test_parallel_hedge_secondary_wins_on_stall() -> None:
    import asyncio as _aio

    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=60.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
        enable_parallel_hedge=True,
        parallel_hedge_delay_seconds=0.05,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_stream(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")

        async def slow():
            await _aio.sleep(5.0)
            yield b"data: late\n\n"

        async def fast():
            yield b'data: {"choices": [{"delta": {"content": "hedged"}}]}\n\n'

        if model == "model-a":
            return 200, slow(), {"content-type": "text/event-stream"}, _key()
        return 200, fast(), {"content-type": "text/event-stream"}, _key(1)

    upstream = AsyncMock()
    upstream.stream = fake_stream
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._client_for = lambda m: (upstream, m)  # type: ignore[method-assign]
    # Provider-diverse pair so the hedge is eligible.
    ex._provider_id_for = lambda m: {"model-a": "p1", "model-b": "p2"}[m]  # type: ignore[method-assign]

    result = await ex.execute_stream(
        "/chat/completions", {"messages": []}, _auto(["model-a", "model-b"])
    )
    assert result.status_code == 200
    assert result.model == "model-b"
    assert ex.stats.hedge_attempts == 1
    assert ex.stats.hedge_wins == 1
    body = b"".join([chunk async for chunk in result.byte_iter])
    assert b"hedged" in body


# ── T12: recovery pool never crashes on empty chains (review critical) ──


def test_recovery_pool_empty_chain_no_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = set()
    ex = FallbackExecutor(AsyncMock(), reg, settings)
    monkeypatch.setattr(ex, "_chain", lambda d, had_tools=False: [])
    monkeypatch.setattr(ex, "_heal_empty_chain", lambda *a, **k: [])
    import potato.resilience as _res

    monkeypatch.setattr(_res, "emergency_chain", lambda *a, **k: [])
    fresh, chain_retry = ex._recovery_pool(
        _auto([]), had_tools=False, tried=set(), allow_any_live=False
    )
    assert fresh == []
    assert chain_retry == []


# ── T13: key-acquire wait surfaces as capacity, fast (review high) ──


@pytest.mark.asyncio
async def test_acquire_timeout_surfaces_capacity() -> None:
    import time as _time

    from potato.upstream import KeyPoolExhausted, UpstreamClient

    pool = KeyPool(api_keys=["k"])
    pool._keys[0].cooldown_until = _time.monotonic() + 60.0
    client = UpstreamClient(base_url="http://localhost:1", pool=pool)
    t0 = _time.monotonic()
    with pytest.raises(KeyPoolExhausted):
        await client.request_json("POST", "/chat/completions", acquire_timeout=0.2)
    assert _time.monotonic() - t0 < 5.0


# ── T14: generic RuntimeError in execute_json advances to next fallback candidate (RCA) ──


@pytest.mark.asyncio
async def test_generic_runtime_error_advances_fallback() -> None:
    """RCA: upstream errors wrapped in RuntimeError('upstream failed after retries: ...')
    must advance through fallback chain rather than raising fatal exception to openai.py.
    """
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=3,
        request_deadline_seconds=30.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    calls: list[str] = []

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        calls.append(model)
        if model == "model-a":
            # UpstreamClient wraps httpx.ReadTimeout in RuntimeError:
            raise RuntimeError("upstream failed after retries: The read operation timed out")
        return 200, {"choices": [{"message": {"content": "ok from b"}}]}, {}, _key(1)

    upstream = AsyncMock()
    upstream.request_json = fake_json
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    decision = _auto(["model-a", "model-b"])

    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    assert calls == ["model-a", "model-b"]
    assert ex.stats.fallback_advances >= 1


# ── T15: HTTP 503 from model cools model-tier without opening provider breaker (D3) ──


@pytest.mark.asyncio
async def test_http_503_cools_model_without_opening_provider_breaker() -> None:
    """HTTP 503 is a model-tier overload condition, not a provider transport death.
    Provider breaker must remain CLOSED.
    """
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=3,
        request_deadline_seconds=30.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        if model == "model-a":
            return 503, {"error": {"message": "Model overloaded"}}, {}, _key()
        return 200, {"choices": [{"message": {"content": "b ok"}}]}, {}, _key(1)

    upstream = AsyncMock()
    upstream.request_json = fake_json
    cb = ProviderCircuitBreaker(failure_threshold=2)  # Low threshold
    ex = FallbackExecutor(upstream, reg, settings, hub=_hub_stub(cb, {"model-a": upstream, "model-b": upstream}))
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._provider_id_for = lambda m: "nim"  # type: ignore[method-assign]

    decision = _auto(["model-a", "model-b"])
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    # Model A is cooled down:
    assert cb.is_model_on_cooldown("model-a")
    # But provider breaker is STILL CLOSED!
    assert cb.state("nim") == BreakerState.CLOSED


# ── T16: Hedged secondary win does not increment provider transport failure count ──


@pytest.mark.asyncio
async def test_hedged_secondary_win_does_not_fail_provider_transport() -> None:
    """When a secondary hedged stream completes first, primary is abandoned but primary provider
    breaker must NOT count it as a transport failure.
    """
    import asyncio as _aio

    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=30.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
        enable_parallel_hedge=True,
        parallel_hedge_delay_seconds=0.02,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"p1/model-a", "p2/model-b"}

    async def fake_stream(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")

        async def slow():
            await _aio.sleep(2.0)
            yield b"data: slow\n\n"

        async def fast():
            yield b'data: {"choices": [{"delta": {"content": "fast"}}]}\n\n'

        if "model-a" in model:
            return 200, slow(), {"content-type": "text/event-stream"}, _key()
        return 200, fast(), {"content-type": "text/event-stream"}, _key(1)

    upstream = AsyncMock()
    upstream.stream = fake_stream
    cb = ProviderCircuitBreaker(failure_threshold=1)  # Threshold=1: any transport fail would trip OPEN!
    ex = FallbackExecutor(upstream, reg, settings, hub=_hub_stub(cb, {"p1/model-a": upstream, "p2/model-b": upstream}))
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._client_for = lambda m: (upstream, m)  # type: ignore[method-assign]
    ex._provider_id_for = lambda m: m.split("/")[0]  # type: ignore[method-assign]

    decision = _auto(["p1/model-a", "p2/model-b"])
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "p2/model-b"
    assert ex.stats.hedge_wins == 1
    # p1 circuit breaker MUST NOT be tripped!
    assert cb.state("p1") == BreakerState.CLOSED


# ── T17: Horizontal provider discovery for explicit model ──


def test_horizontal_provider_discovery_for_explicit_model() -> None:
    """When enable_fallback_on_explicit=True, selector discovers same model across providers."""
    from potato.routing.classifier import IntentResult
    from potato.routing.selector import ModelSelector

    settings = Settings(
        nim_api_keys=["k"],
        enable_fallback_on_explicit=True,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {
        "groq/llama-3.3-70b",
        "together/llama-3.3-70b",
        "sambanova/llama-3.3-70b",
        "qwen/qwen-2.5-72b",
    }
    reg.ladder.provider_ids = {"groq", "together", "sambanova", "qwen", "nim"}

    selector = ModelSelector(reg, settings)
    intent_res = IntentResult(intent=Intent.CHAT_FAST, confidence=0.95, rule_id="test")
    decision = selector.resolve("groq/llama-3.3-70b", intent_res)

    assert decision.mode == "passthrough_with_fallback"
    assert decision.chain[0] == "groq/llama-3.3-70b"
    assert "together/llama-3.3-70b" in decision.chain[1:3]
    assert "sambanova/llama-3.3-70b" in decision.chain[1:3]


# ── T18: Robust iter suppresses raw in-band error frames ──


@pytest.mark.asyncio
async def test_robust_iter_suppresses_raw_upstream_error_chunk() -> None:
    """When an upstream stream returns 200 followed by an in-band error frame,
    robust_iter suppresses the raw error chunk and emits a clean OpenAI SSE finish event.
    """
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_error_stream():
        yield b'data: {"id":"1","choices":[{"delta":{"content":"starting"}}]}\n\n'
        yield b'data: {"error":{"message":"Midstream failure","code":"model_error"}}\n\n'

    async def fake_stream(method, path, **kwargs):
        return 200, fake_error_stream(), {"content-type": "text/event-stream"}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]
    ex._client_for = lambda m: (upstream, m)  # type: ignore[method-assign]

    decision = _auto(["model-a"])
    result = await ex.execute_stream("/chat/completions", {"messages": [], "stream": True}, decision)
    assert result.status_code == 200

    chunks = [c async for c in result.byte_iter]
    combined = b"".join(chunks)

    # First chunk was content
    assert b"starting" in combined
    # Raw unparsed error JSON was suppressed and replaced by potato-stream-error
    assert b"potato-stream-error" in combined
    assert b"Midstream failure" in combined
    assert b"data: [DONE]" in combined


# ── T19: Edge Case — Cloudflare 524 Gateway Timeout Advances Fallback ──


@pytest.mark.asyncio
async def test_cloudflare_524_advances_fallback() -> None:
    """When an upstream proxy returns HTTP 524 Gateway Timeout, the fallback engine
    treats it as retryable and advances to the next model in the chain.
    """
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    calls: list[str] = []

    async def fake_request(method, path, *, json_body, **kwargs):
        m = json_body.get("model")
        calls.append(m)
        if m == "model-a":
            return 524, {"error": {"message": "A timeout occurred"}}, {}, _key()
        return 200, {"id": "ok", "choices": [{"message": {"content": "recovered"}}]}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_request
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True
    ex._client_for = lambda m: (upstream, m)

    decision = _auto(["model-a", "model-b"])
    res = await ex.execute_json("/chat/completions", {"messages": []}, decision)

    assert res.status_code == 200
    assert calls == ["model-a", "model-b"]
    assert res.body["choices"][0]["message"]["content"] == "recovered"


# ── T20: Edge Case — HTTP 422 Context Length Overflow Advances Fallback ──


@pytest.mark.asyncio
async def test_context_overflow_422_advances_fallback() -> None:
    """When an upstream model returns HTTP 422 Unprocessable Entity with context overflow,
    the router advances to the next candidate model rather than failing as a client error.
    """
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    calls: list[str] = []

    async def fake_request(method, path, *, json_body, **kwargs):
        m = json_body.get("model")
        calls.append(m)
        if m == "model-a":
            return 422, {"error": {"message": "prompt is too long: exceeds maximum context length of 8192"}}, {}, _key()
        return 200, {"id": "ok", "choices": [{"message": {"content": "big-context-win"}}]}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_request
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True
    ex._client_for = lambda m: (upstream, m)

    decision = _auto(["model-a", "model-b"])
    res = await ex.execute_json("/chat/completions", {"messages": []}, decision)

    assert res.status_code == 200
    assert calls == ["model-a", "model-b"]
    assert res.body["choices"][0]["message"]["content"] == "big-context-win"


# ── T21: Edge Case — Rogue Retry-After Ceiling ──


def test_backoff_sanity_cap_on_rogue_retry_after() -> None:
    """Verify that a rogue or malicious Retry-After header (e.g. 86400s) is capped
    at a reasonable sanity ceiling rather than freezing execution for 24 hours.
    """
    from potato.safety.backoff import compute_backoff_seconds

    # Without max_delay, caps at max(cap, 60.0) = 60s
    delay = compute_backoff_seconds(0, cap=16.0, retry_after=86400.0)
    assert delay <= 75.0  # 60s + max 20% jitter = 72.0s

    # With max_delay=16.0, caps at 16.0s
    capped = compute_backoff_seconds(0, cap=16.0, retry_after=86400.0, max_delay=16.0)
    assert capped <= 16.0


# ── T22: Edge Case — Circuit Breaker Model Cooldown Eviction ──


def test_circuit_breaker_model_cooldown_eviction() -> None:
    """Verify that expired model cooldowns are automatically pruned to prevent memory leaks."""
    cb = ProviderCircuitBreaker()
    # Add an already expired cooldown
    cb._model_cooldowns["expired-model"] = time.monotonic() - 10.0
    cb._model_cooldowns["active-model"] = time.monotonic() + 100.0

    # is_model_on_cooldown checks and prunes expired
    assert not cb.is_model_on_cooldown("expired-model")
    assert "expired-model" not in cb._model_cooldowns
    assert cb.is_model_on_cooldown("active-model")


# ── T23: Edge Case — Non-dict JSON Request Rejection (HTTP 400 vs 500) ──


@pytest.mark.asyncio
async def test_non_dict_json_request_returns_400() -> None:
    """Sending a non-dict JSON body (e.g. a list [1, 2, 3]) must return 400 Bad Request,
    not raise an unhandled 500 AttributeError on body.get().
    """
    from unittest.mock import AsyncMock, MagicMock
    from starlette.requests import Request
    from potato.routes.openai import _chat_like

    async def fake_receive():
        return {
            "type": "http.request",
            "body": b"[1, 2, 3]",
            "more_body": False,
        }

    app = MagicMock()
    scope = {
        "type": "http",
        "app": app,
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "state": {"request_id": "req-test-non-dict"},
    }
    req = Request(scope, fake_receive)
    # Mock minimal app.state
    req.app.state.guard = MagicMock()
    req.app.state.upstream = AsyncMock()

    resp = await _chat_like(req, upstream_path="/chat/completions")
    assert resp.status_code == 400
    import json
    data = json.loads(resp.body.decode("utf-8"))
    assert data["error"]["code"] == "invalid_json"


# ── T24: Edge Case — Claude SSE Stream Emits Anthropic Error Frame ──


@pytest.mark.asyncio
async def test_claude_sse_stream_translates_error_frame() -> None:
    """When an upstream stream fails mid-stream with an error event, Claude SSE translation
    must emit an Anthropic 'event: error' frame and not complete with end_turn.
    """
    from fastapi.responses import StreamingResponse
    from potato.routes.claude import transform_openai_to_anthropic_sse_stream

    async def fake_openai_chunks():
        # First chunk: text
        yield b'data: {"id":"1","choices":[{"delta":{"content":"Hi"}}]}\n\n'
        # Second chunk: error event
        yield b'data: {"error":{"message":"Mid-stream GPU collapse","type":"server_error"}}\n\n'
        yield b'data: [DONE]\n\n'

    stream_resp = StreamingResponse(fake_openai_chunks())
    events: list[str] = []
    async for chunk in transform_openai_to_anthropic_sse_stream(stream_resp, "claude-3-5-sonnet"):
        events.append(chunk.decode("utf-8"))

    combined = "".join(events)
    assert "event: error" in combined
    assert "Mid-stream GPU collapse" in combined
    assert "end_turn" not in combined


# ── T25: Edge Case — Responses API SSE Translates Stream Error to response.failed ──


@pytest.mark.asyncio
async def test_responses_sse_translates_error_to_failed_event() -> None:
    """When an upstream stream fails with an error or finish_reason=error, Responses API
    must emit a 'response.failed' event frame instead of 'response.completed'.
    """
    from fastapi.responses import StreamingResponse
    from potato.routes.responses import transform_chat_sse_to_responses_sse

    async def fake_error_chunks():
        yield b'data: {"choices":[{"delta":{},"finish_reason":"error"}]}\n\n'
        yield b'data: [DONE]\n\n'

    stream_resp = StreamingResponse(fake_error_chunks())
    events: list[str] = []
    async for chunk in transform_chat_sse_to_responses_sse(stream_resp, "gpt-4o"):
        events.append(chunk.decode("utf-8"))

    combined = "".join(events)
    assert "response.failed" in combined
    assert "response.completed" not in combined


# ── T26: Edge Case — Plain String HTTP 400 Retryable Error Advances Fallback ──


@pytest.mark.asyncio
async def test_plain_string_400_retryable_error_advances_fallback() -> None:
    """When an upstream proxy returns a raw string HTTP 400 (e.g. 'Upstream request failed'),
    the fallback executor advances rather than treating it as a non-retryable client error.
    """
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    calls: list[str] = []

    async def fake_request(method, path, *, json_body, **kwargs):
        m = json_body.get("model")
        calls.append(m)
        if m == "model-a":
            # Plain string body from reverse proxy
            return 400, "Bad Request: upstream request failed due to backend timeout", {}, _key()
        return 200, {"id": "ok", "choices": [{"message": {"content": "rescued"}}]}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_request
    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True
    ex._client_for = lambda m: (upstream, m)

    decision = _auto(["model-a", "model-b"])
    res = await ex.execute_json("/chat/completions", {"messages": []}, decision)

    assert res.status_code == 200
    assert calls == ["model-a", "model-b"]
    assert res.body["choices"][0]["message"]["content"] == "rescued"

