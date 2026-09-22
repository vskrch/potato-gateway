"""Ordered model fallback execution (separate from key rotation)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from potato.routing.selector import RouteDecision
from potato.safety.backoff import sleep_backoff
from potato.upstream import KeyPoolExhausted, parse_retry_after

if TYPE_CHECKING:
    from potato.analytics.models import TraceSpan
    from potato.balancer import KeyStats
    from potato.catalog.registry import ModelRegistry
    from potato.config import Settings
    from potato.upstream import UpstreamClient

logger = logging.getLogger(__name__)

SpanCallback = Callable[["TraceSpan"], None]

# D3: model-tier statuses cool only the model, never the provider transport
# breaker. Everything else >= 400 that is retryable is transport-class.
MODEL_TIER_STATUSES = frozenset({400, 401, 403, 404, 405, 408, 413, 422, 429, 503, 529})


@dataclass
class UpstreamResult:
    status_code: int
    body: Any
    headers: dict[str, str]
    key: KeyStats | None
    model: str
    fallback_index: int
    decision: RouteDecision
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    upstream_ms: float | None = None
    provider_id: str | None = None


@dataclass
class StreamResult:
    status_code: int
    byte_iter: AsyncIterator[bytes]
    headers: dict[str, str]
    key: KeyStats | None
    model: str
    fallback_index: int
    decision: RouteDecision
    upstream_ttft_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    provider_id: str | None = None
    # Mutable usage bag updated as SSE chunks are scanned (stream may finish after return)
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
    )
    # True when robust_iter detected a mid-stream failure and emitted error events
    stream_failed: bool = False


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RoutingStats:
    intents_total: dict[str, int] = field(default_factory=dict)
    models_total: dict[str, int] = field(default_factory=dict)
    model_tokens: dict[str, TokenStats] = field(default_factory=dict)
    key_tokens: dict[str, TokenStats] = field(default_factory=dict)
    fallback_advances: int = 0
    # P2-4 SLO counters: recovery entered vs chain exhausted (D1's smoking gun:
    # chain_exhausted should be ~0 once the recovery reserve ships).
    recovery_entered: int = 0
    chain_exhausted: int = 0
    hedge_attempts: int = 0
    hedge_wins: int = 0
    # Adaptive ranking: track last 50 requests' advance status (NMK-304)
    _recent_advances: list[bool] = field(default_factory=list)
    _max_advances_track: int = 50

    def record(self, intent: str, model: str, advanced: bool) -> None:
        self.intents_total[intent] = self.intents_total.get(intent, 0) + 1
        self.models_total[model] = self.models_total.get(model, 0) + 1
        if advanced:
            self.fallback_advances += 1
        self._recent_advances.append(advanced)
        if len(self._recent_advances) > self._max_advances_track:
            self._recent_advances = self._recent_advances[-self._max_advances_track :]

    def should_rerank(self) -> bool:
        """True when >30% of recent requests advanced → rankings may be stale."""
        if len(self._recent_advances) < 20:
            return False
        return sum(self._recent_advances) / len(self._recent_advances) > 0.30

    def record_tokens(self, model: str, key_id: str | None, in_tok: int, out_tok: int) -> None:
        if model not in self.model_tokens:
            self.model_tokens[model] = TokenStats()
        self.model_tokens[model].prompt_tokens += in_tok
        self.model_tokens[model].completion_tokens += out_tok

        if key_id:
            if key_id not in self.key_tokens:
                self.key_tokens[key_id] = TokenStats()
            self.key_tokens[key_id].prompt_tokens += in_tok
            self.key_tokens[key_id].completion_tokens += out_tok


def _is_model_not_found(status: int, body: Any) -> bool:
    if status == 404:
        return True
    text = ""
    if isinstance(body, dict):
        err = body.get("error")
        text = str(err.get("message") or "") if isinstance(err, dict) else str(body)
    elif isinstance(body, str):
        text = body
    low = text.lower()
    return any(
        s in low for s in ("model not found", "unknown model", "does not exist", "invalid model")
    )


def _is_retryable_model_error(status: int, body: Any) -> bool:
    if status in {401, 403, 405, 408, 429, 500, 502, 503, 504}:
        return True
    if _is_model_not_found(status, body):
        return True
    # Tools unsupported → try next model
    if status == 400 and isinstance(body, dict):
        msg = str((body.get("error") or {}).get("message") or "").lower()
        if "tool" in msg and ("not support" in msg or "unsupported" in msg):
            return True
        # Providers that wrap server errors in 400 responses
        _retryable_phrases = (
            "upstream request failed",
            "error from provider",
            "internal error",
            "service unavailable",
            "request failed",
            "bad gateway",
            "gateway timeout",
        )
        if any(phrase in msg for phrase in _retryable_phrases):
            logger.info(
                "retryable 400 detected: status=%s msg=%s",
                status,
                msg[:200],
            )
            return True
    return status in {400, 413} and _is_context_overflow_message(_body_message(body))


def _body_message(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or "")
        return str(body)
    if isinstance(body, str):
        return body
    return ""


def _is_context_overflow_message(msg: str) -> bool:
    low = msg.lower()
    if any(
        s in low
        for s in (
            "context length",
            "context window",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "prompt is too long",
        )
    ):
        return True
    return bool(re.search(r"context.*exceed|exceeds.*context|maximum.*tokens", low))


def _is_non_retryable_client_error(status: int, body: Any) -> bool:
    # 401/403 are retryable across providers (keys already rotated within the
    # current provider by UpstreamClient); only treat 400/422 as hard-stop
    # unless the body signals a model/context issue.
    if status in {400, 422}:
        return not _is_retryable_model_error(status, body)
    return False


def _analyze_success_body(
    body: Any, *, had_tools: bool, path: str = "/chat/completions"
) -> tuple[bool, bool | None]:
    """
    Returns (empty_reply, tool_ok).
    tool_ok is None when tools were not requested.

    Schema-aware: chat (message), text completions (text), responses (output),
    embeddings (data[].embedding).
    """
    if not isinstance(body, dict):
        # Non-dict 2xx bodies (HTML error pages, text) are unusable — treat
        # as an empty reply so the chain advances instead of serving garbage.
        return True, False if had_tools else None

    path_l = (path or "").lower()
    is_completions = path_l.endswith("/completions") and "chat" not in path_l
    is_responses = "/responses" in path_l
    is_embeddings = "/embeddings" in path_l

    # Embeddings: data[].embedding must be a non-empty vector
    if is_embeddings:
        data = body.get("data")
        if not isinstance(data, list) or not data:
            return True, None
        for item in data:
            if not isinstance(item, dict):
                return True, None
            emb = item.get("embedding")
            if not isinstance(emb, list) or len(emb) == 0:
                return True, None
        return False, None

    # Text completions: choices[0].text
    if is_completions:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return True, None
        ch0 = choices[0] if isinstance(choices[0], dict) else {}
        text = ch0.get("text")
        empty = text in (None, "")
        return empty, None

    # Responses API: output / output_text
    if is_responses:
        if body.get("output_text"):
            return False, None if not had_tools else True
        output = body.get("output")
        if isinstance(output, list) and output:
            # Any non-empty content/text in output items = success
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str) and content:
                    return False, None if not had_tools else True
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and (
                            part.get("text") or part.get("type") == "function_call"
                        ):
                            return False, None if not had_tools else True
                if item.get("type") in {"function_call", "tool_call"}:
                    return False, True if had_tools else None
            return True, False if had_tools else None
        return True, False if had_tools else None

    # Chat completions
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return True, False if had_tools else None
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return True, False if had_tools else None
    content = msg.get("content")
    tool_calls = msg.get("tool_calls") or msg.get("function_call")
    empty = content in (None, "", []) and not tool_calls
    if not had_tools:
        return empty, None
    if tool_calls:
        return empty, True
    # Tools requested but none returned — soft fail signal
    return empty, False


class FallbackExecutor:
    def __init__(
        self,
        upstream: UpstreamClient,
        registry: ModelRegistry,
        settings: Settings,
        stats: RoutingStats | None = None,
        hub: Any | None = None,
        span_callback: SpanCallback | None = None,
        rl_engine: Any | None = None,
    ) -> None:
        self.upstream = upstream
        self.registry = registry
        self.settings = settings
        self.stats = stats or RoutingStats()
        self.hub = hub
        self.rl_engine = rl_engine
        self._span_cb = span_callback or None
        # Prefer contextvar collector so concurrent requests don't share state
        from potato.analytics.context import collect_span

        if self._span_cb is None:
            self._span_cb = collect_span

    def set_span_callback(self, cb: SpanCallback | None) -> None:
        self._span_cb = cb
        if self._span_cb is None:
            from potato.analytics.context import collect_span

            self._span_cb = collect_span

    def _emit_span(self, span: Any) -> None:
        if self._span_cb is None:
            return
        try:
            self._span_cb(span)
        except Exception:
            logger.debug("span callback failed", exc_info=True)

    def _record_rl_feedback(
        self,
        decision: RouteDecision,
        model: str,
        *,
        success: bool,
        status_code: int = 200,
        latency: float | None = None,
        tool_ok: bool | None = None,
        empty_reply: bool = False,
    ) -> None:
        """Feed the LinUCB bandit one (model, x, reward) sample from execution.

        ponytail: no-op when rl_engine is absent (tests / routing off).
        Reward is computed from the same multi-signal feedback the health
        store already uses — TTFB, status, tool validity — via
        ``calculate_composite_reward``.  Bounded to one update per request
        per model so streaming + JSON paths don't double-count.
        """
        rl = self.rl_engine
        x = getattr(decision, "feature_vector", None)
        if rl is None or not x or len(x) != 12:
            return
        try:
            from potato.routing.rl_rewards import calculate_composite_reward

            ttfb = latency if latency is not None and latency > 0 else None
            reward = calculate_composite_reward(
                success=success,
                status_code=status_code,
                ttfb_seconds=ttfb,
                tool_ok=tool_ok,
                empty_reply=empty_reply,
            )
            rl.record_feedback(model, x, reward)
        except Exception:
            logger.debug("RL feedback update failed", exc_info=True)

    def _record_outcome(
        self,
        decision: RouteDecision,
        model: str,
        key_id: str | None = None,
        *,
        success: bool,
        latency: float | None = None,
        status_code: int | None = None,
        unavailable: bool = False,
        tokens: int | None = None,
        intent: str | None = None,
        empty_reply: bool = False,
        had_tools: bool = False,
        tool_ok: bool | None = None,
    ) -> None:
        """Record health outcome + RL bandit feedback in one shot.

        Wraps registry.record_outcome so every execution path feeds the
        LinUCB engine the same (model, x, reward) signal it already feeds
        the health store.  Args mirror registry.record_outcome exactly.
        """
        self.registry.record_outcome(
            model,
            key_id,
            success=success,
            latency=latency,
            status_code=status_code,
            unavailable=unavailable,
            tokens=tokens,
            intent=intent,
            empty_reply=empty_reply,
            had_tools=had_tools,
            tool_ok=tool_ok,
        )
        self._record_rl_feedback(
            decision,
            model,
            success=success,
            status_code=status_code or 200,
            latency=latency,
            tool_ok=tool_ok,
            empty_reply=empty_reply,
        )

    def _provider_id_for(self, model: str) -> str | None:
        if self.hub is None:
            return None
        try:
            from potato.catalog.providers import split_provider_model

            pid, _ = split_provider_model(model, self.hub.provider_ids, default_provider="nim")
            return pid
        except Exception:
            return None

    def _circuit_fail(
        self,
        provider_id: str | None,
        *,
        is_transport: bool = True,
        model_id: str | None = None,
    ) -> None:
        if not provider_id or self.hub is None:
            return
        cb = getattr(self.hub, "circuit_breaker", None)
        if cb is not None:
            cb.fail(provider_id, is_transport=is_transport, model_id=model_id)

    def _record_circuit(
        self,
        provider_id: str | None,
        *,
        model: str,
        status: int | None = None,
        transport: bool = False,
    ) -> None:
        """Two-tier circuit classification (D3).

        transport=True → provider transport breaker (5xx / timeouts /
        transport exceptions). status in MODEL_TIER_STATUSES → model-tier
        cooldown only; the provider breaker is untouched.
        """
        if transport:
            self._circuit_fail(provider_id, model_id=model)
            return
        if (
            status is not None
            and status in MODEL_TIER_STATUSES
            and provider_id
            and self.hub is not None
        ):
            cb = getattr(self.hub, "circuit_breaker", None)
            if cb is not None:
                cb.fail(provider_id, is_transport=False, model_id=model)
                return
        if status is not None and status >= 500:
            self._circuit_fail(provider_id, model_id=model)

    def _model_on_cooldown(self, model: str) -> bool:
        """True when the breaker's model tier has cooled this model (D3/P0-3c)."""
        if self.hub is None:
            return False
        cb = getattr(self.hub, "circuit_breaker", None)
        if cb is None:
            return False
        try:
            pid = self._provider_id_for(model)
            for candidate in {model, f"{pid}/{model}"} if pid else {model}:
                if cb.is_model_on_cooldown(candidate):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _phase_deadlines(deadline: float, now: float) -> dict[str, float]:
        """P1-4: split the remaining budget into primary / recovery / guard.

        Recovery runs until 85% of the deadline; the final slice is reserved
        for the structured error envelope + logging.
        """
        total = max(0.0, deadline - now)
        return {
            "primary": now + total * 0.60,
            "recovery": deadline - max(2.0, total * 0.15),
        }

    def _recovery_pool(
        self,
        decision: RouteDecision,
        *,
        had_tools: bool,
        tried: set[str],
        allow_any_live: bool = True,
    ) -> tuple[list[str], list[str]]:
        """D5a: widest-net recovery candidates (any-live first, then chain).

        Returns (fresh, chain_retry) where fresh excludes already-tried models.
        allow_any_live=False preserves the explicit-request invariant (an
        explicit model request is never served by a different model).
        """
        pools: list[list[str]] = []
        if allow_any_live:
            try:
                pools.append(
                    self._any_available_live_models(
                        had_tools=had_tools,
                        intent=decision.intent.value,
                        allowed_models=list(getattr(decision, "allowed_models", None) or []),
                        free_only=(str(getattr(decision, "auto_tier", "") or "").lower() == "free"),
                    )
                )
            except Exception:
                logger.debug("recovery pool build failed", exc_info=True)
        try:
            chain_retry = self._chain(decision, had_tools=had_tools) or []
        except Exception:
            logger.debug("recovery chain rebuild failed", exc_info=True)
            chain_retry = []
        if not chain_retry:
            try:
                chain_retry = self._heal_empty_chain(
                    decision,
                    max_n=self._max_n_for_intent(decision.intent.value),
                    disabled=getattr(self.registry, "disabled_models", None) or set(),
                    had_tools=had_tools,
                )
            except Exception:
                try:
                    from potato.resilience import emergency_chain

                    chain_retry = emergency_chain(
                        self.registry,
                        intent=decision.intent.value,
                        max_n=self._max_n_for_intent(decision.intent.value),
                    )
                except Exception:
                    chain_retry = []
        pools.append(chain_retry)
        seen, fresh = set(tried), []
        for pool in pools:
            for m in pool:
                if m.lower() not in seen:
                    seen.add(m.lower())
                    fresh.append(m)
        return fresh, chain_retry

    def _clear_model_cooldowns(self, models: list[str]) -> None:
        """D4: scoped cooldown relief — only models about to be retried."""
        health = getattr(self.registry, "health", None)
        if health is None:
            return
        for m in models:
            h = health._by_model.get(m)
            if h is not None:
                h.cooldown_until = 0.0

    def _enforce_provider_diversity(self, chain: list[str]) -> list[str]:
        """P2-2: no more than 2 consecutive same-provider candidates; ensure
        >= 2 distinct providers in the first 4 slots whenever available."""
        if len(chain) <= 2:
            return chain
        providers = [self._provider_id_for(m) or "" for m in chain]
        if len(set(providers)) <= 1:
            return chain
        out: list[str] = []
        out_pids: list[str] = []
        pending = list(zip(chain, providers))
        # Greedy: prefer a different provider than the last two picks.
        while pending:
            pick_idx = None
            if len(out_pids) >= 2 and out_pids[-1] == out_pids[-2]:
                for i, (_, p) in enumerate(pending):
                    if p != out_pids[-1]:
                        pick_idx = i
                        break
            if pick_idx is None:
                pick_idx = 0
            m, p = pending.pop(pick_idx)
            out.append(m)
            out_pids.append(p)
        # Guarantee >= 2 providers in the first 4 when available.
        first4 = set(out_pids[:4])
        if len(first4) < 2:
            for i in range(4, len(out)):
                if out_pids[i] not in first4:
                    m = out.pop(i)
                    p = out_pids.pop(i)
                    out.insert(1, m)
                    out_pids.insert(1, p)
                    break
        return out

    def _circuit_succeed(
        self,
        provider_id: str | None,
        *,
        model_id: str | None = None,
    ) -> None:
        if not provider_id or self.hub is None:
            return
        cb = getattr(self.hub, "circuit_breaker", None)
        if cb is not None:
            cb.succeed(provider_id, model_id=model_id)

    def _make_upstream_span(
        self,
        *,
        model: str,
        t0: float,
        status: int | None = None,
        success: bool = True,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        span_type: str = "upstream",
    ) -> Any:
        from potato.analytics.models import TraceSpan

        ended = time.perf_counter()
        return TraceSpan(
            span_type=span_type,
            model_id=model,
            provider_id=self._provider_id_for(model),
            started_at=t0,
            ended_at=ended,
            duration_ms=(ended - t0) * 1000,
            status_code=status,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )

    def _client_for(self, model: str) -> tuple[Any, str]:
        """Return (upstream_client, upstream_model_id) for this namespaced model."""
        # Use original-case id for upstream round-trip (case-sensitive providers)
        original = self.registry.original_id(model)
        if self.hub is not None:
            client, _pid, upstream_mid = self.hub.client_for_model(original)
            return client, upstream_mid
        return self.upstream, original

    def _provider_available(self, model: str) -> bool:
        if self.hub is None:
            return True
        try:
            from potato.catalog.providers import split_provider_model

            pid, _ = split_provider_model(model, self.hub.provider_ids, default_provider="nim")
            return self.hub.has_runtime(pid)
        except Exception:
            logger.exception("provider availability check failed for model %s", model)
            return False

    def _any_available_live_models(
        self,
        *,
        had_tools: bool = False,
        intent: str | None = None,
        allowed_models: list[str] | None = None,
        free_only: bool = False,
    ) -> list[str]:
        """Every live model whose provider has a runtime — widest possible net.

        Used by the graceful fallback when the intent-aware chain is exhausted.
        Prefers models that support tools when tools were requested, but falls
        back to all live models rather than returning empty (serving > 503).
        Hard routing constraints (``allowed_models`` / free-only) are
        preserved so fallback never serves models the caller excluded, and the
        result is quality + health ranked so fallback still prefers the
        strongest responding models instead of arbitrary set order.
        """
        registry = self.registry
        active = list(getattr(registry, "active_live_ids", lambda: set())() or set())
        if not active:
            active = list(getattr(registry, "live_ids", set()) or set())
        available = [m for m in active if self._provider_available(m)]
        if allowed_models or free_only:
            from potato.routing.auto_router import filter_chain

            available = filter_chain(
                available, allowed_models=allowed_models or None, free_only=free_only
            )
        if not available:
            return []
        if had_tools and hasattr(registry, "ladder"):
            caps = getattr(registry.ladder, "capabilities", {})
            tool_ok = [
                m for m in available if (caps.get(m) or {}).get("supports_tools") is not False
            ]
            non_ok = [m for m in available if m not in set(tool_ok)]
            available = tool_ok + non_ok
        # Quality + health ranking: graceful fallback must still prefer the
        # strongest responding models, not insertion/set order.
        try:
            ranked = registry.health_reorder(available, intent=intent or "chat_fast")
            if ranked:
                return ranked
        except Exception:
            logger.debug(
                "graceful fallback quality ranking failed — using insertion order",
                exc_info=True,
            )
        return available

    def _best_exploration_candidate(
        self,
        current: list[str],
        *,
        intent: str,
        allowed_models: list[str],
        free_only: bool,
    ) -> str | None:
        """Highest-UCB under-sampled live model not already leading the chain.

        Anti-celebrity exploration: models with few samples get the largest
        UCB1 bonus, so a challenger that the ladder head has starved finally
        gets sampled. Quality gate: the candidate must score >=
        ``rl_exploration_min_quality_ratio`` × the best current candidate —
        exploration never serves a model far below the proven head.
        """
        import math

        registry = self.registry
        try:
            ladder = getattr(registry, "ladder", None)
            snap = getattr(ladder, "_ladders", {}).get((intent, "default"))
            scores = snap.scores if snap is not None else {}
        except Exception:
            scores = {}
        top_score = max((scores.get(m, 0.0) for m in current), default=0.0)
        min_ratio = float(
            getattr(self.settings, "rl_exploration_min_quality_ratio", 0.5) or 0.5
        )
        floor = top_score * min_ratio if top_score > 0 else None

        from potato.routing.auto_router import filter_chain

        candidates = list(getattr(registry, "active_live_ids", lambda: set())() or set())
        candidates = [m for m in candidates if self._provider_available(m)]
        candidates = [m for m in candidates if m not in current[:2]]
        if allowed_models or free_only:
            candidates = filter_chain(
                candidates, allowed_models=allowed_models or None, free_only=free_only
            )
        if floor is not None:
            candidates = [m for m in candidates if scores.get(m, 0.0) >= floor]

        total_n = registry.learning.total_requests(intent)
        blend_n = int(getattr(self.settings, "thompson_blend_n", 12) or 12)
        c = float(getattr(self.settings, "ucb_exploration_c", 5.0) or 5.0)
        best: str | None = None
        best_ucb = -1.0
        best_quality = -1.0
        for m in candidates:
            n = registry.learning.model_requests(intent, m)
            if n >= blend_n:
                continue  # already sufficiently sampled — not a challenger
            if total_n < 2:
                ucb = c * 2.0
            elif n == 0:
                ucb = c * math.sqrt(math.log(total_n + 1))
            else:
                ucb = c * math.sqrt(math.log(total_n + 1) / n)
            quality = scores.get(m, 0.0)
            if ucb > best_ucb or (ucb == best_ucb and quality > best_quality):
                best_ucb = ucb
                best_quality = quality
                best = m
        return best

    async def _try_models(
        self,
        models: list[str],
        *,
        body: dict[str, Any],
        path: str,
        decision: RouteDecision,
        deadline: float,
        forward_headers: dict[str, str] | None,
        preferred_key_id: str | None,
        had_tools: bool,
        chain_len: int,
        start_idx: int,
        last: UpstreamResult,
        max_attempts: int,
    ) -> UpstreamResult:
        """Try a list of models sequentially; return first success or last failure."""
        import asyncio as _aio

        import httpx

        from potato.compat import wrap_upstream_error as _wue

        attempts = 0
        for offset, model in enumerate(models):
            if attempts >= max_attempts:
                break
            remaining = deadline - time.monotonic()
            if remaining < 3.0:
                break
            try:
                client, upstream_mid = self._client_for(model)
            except Exception:
                # D2: circuit-open skip / unconfigured provider / capacity —
                # never a transport failure. Just advance.
                continue
            attempt_body = {**body, "model": upstream_mid}
            # Per-model reasoning_effort normalization (resilience: a reasoning
            # head failing over to a non-reasoning model strips the field).
            from potato.compat import normalize_reasoning_effort as _nre

            attempt_body = _nre(
                attempt_body,
                routed_model=model,
                registry=self.registry,
                default_effort=getattr(self.settings, "default_reasoning_effort", ""),
            )
            if hasattr(self.registry, "ladder"):
                rec = self.registry.ladder.model_recommendations(model)
                if rec:
                    ml = rec.get("max_tokens_limit")
                    if ml and not attempt_body.get("max_tokens"):
                        attempt_body["max_tokens"] = ml
                    elif ml and attempt_body.get("max_tokens"):
                        attempt_body["max_tokens"] = min(attempt_body["max_tokens"], ml)
            pid = self._provider_id_for(model)
            t_attempt = time.perf_counter()
            try:
                budget = self._attempt_budget_for(decision.intent.value, remaining)
                status, resp_body, headers, key = await _aio.wait_for(
                    client.request_json(
                        "POST",
                        path,
                        json_body=attempt_body,
                        forward_headers=forward_headers,
                        preferred_key_id=preferred_key_id,
                        max_retries=1,
                        acquire_timeout=min(2.0, budget),
                    ),
                    timeout=budget,
                )
            except (TimeoutError, RuntimeError, httpx.HTTPError, OSError) as exc:
                attempts += 1
                # D3: capacity (pool exhaustion) cools the model/key tier only.
                if isinstance(exc, KeyPoolExhausted):
                    if self.hub is not None:
                        cb = getattr(self.hub, "circuit_breaker", None)
                        if cb is not None:
                            cb.fail(pid or "", is_transport=False, model_id=model)
                else:
                    self._circuit_fail(pid, model_id=model)
                self._record_outcome(
                    decision,
                    model,
                    None,
                    success=False,
                    status_code=503,
                    intent=decision.intent.value,
                )
                continue

            attempts += 1
            lat = (time.perf_counter() - t_attempt) * 1000
            success = 200 <= status < 300
            if success:
                self._circuit_succeed(pid, model_id=model)
                empty_reply, tool_ok = _analyze_success_body(
                    resp_body, had_tools=had_tools, path=path
                )
                if empty_reply or (had_tools and tool_ok is False):
                    # Soft-fail: 2xx with an unusable body (HTML/empty, or no
                    # tool call when tools were requested) must not be served.
                    # D3: quality signal → model tier only, not transport.
                    self._record_circuit(pid, model=model, status=422)
                    self._record_outcome(
                        decision,
                        model,
                        key.key_id if key else None,
                        success=False,
                        latency=(time.perf_counter() - t_attempt) / 1000,
                        status_code=status,
                        intent=decision.intent.value,
                        empty_reply=empty_reply,
                        had_tools=had_tools,
                        tool_ok=tool_ok,
                    )
                    last = UpstreamResult(
                        status_code=status,
                        body=_wue(resp_body, status=status),
                        headers=headers,
                        key=key,
                        model=model,
                        fallback_index=start_idx + offset,
                        decision=decision,
                        upstream_ms=lat,
                        provider_id=pid,
                    )
                    continue
                pt = ct = cached = 0
                if isinstance(resp_body, dict):
                    usage = resp_body.get("usage") or {}
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    cached = int(
                        usage.get("cached_tokens")
                        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                        or 0
                    )
                self._record_outcome(
                    decision,
                    model,
                    key.key_id if key else None,
                    success=True,
                    latency=lat / 1000,
                    tokens=pt + ct,
                    status_code=status,
                    intent=decision.intent.value,
                )
                self.stats.record(decision.intent.value, model, advanced=True)
                self.stats.record_tokens(model, key.key_id if key else None, pt, ct)
                if isinstance(resp_body, dict) and "model" in resp_body:
                    resp_body = {**resp_body, "model": model}
                return UpstreamResult(
                    status_code=status,
                    body=resp_body,
                    headers=headers,
                    key=key,
                    model=model,
                    fallback_index=chain_len + offset,
                    decision=decision,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    cached_tokens=cached,
                    upstream_ms=lat,
                    provider_id=pid,
                )
            # D3: model-tier statuses cool only the model; transport-class
            # failures trip the provider breaker.
            self._record_circuit(pid, model=model, status=status)
            self._record_outcome(
                decision,
                model,
                key.key_id if key else None,
                success=False,
                status_code=status,
                intent=decision.intent.value,
            )
            last = UpstreamResult(
                status_code=status,
                body=_wue(resp_body, status=status),
                headers=headers,
                key=key,
                model=model,
                fallback_index=start_idx + offset,
                decision=decision,
                upstream_ms=lat,
                provider_id=pid,
            )
        return last

    def _make_deadline(self, intent: str | None = None) -> float:
        """Shared request deadline — both execute_json and execute_stream use this.

        Per-intent overrides via ``intent_deadline_seconds`` config let
        long_horizon / reasoning run far longer than chat_fast.
        """
        base = float(getattr(self.settings, "request_deadline_seconds", 120.0) or 120.0)
        if intent:
            overrides = getattr(self.settings, "intent_deadline_seconds", {}) or {}
            base = float(overrides.get(intent, base))
        return time.monotonic() + base

    def _max_n_for_intent(self, intent: str) -> int:
        """Per-intent fallback cap from config (replaces coding_max_fallbacks).

        The universal ``max_model_fallbacks`` remains the hard ceiling: the
        effective budget is min(per-intent cap, universal cap) so operators
        can lower MAX_MODEL_FALLBACKS below a per-intent default and every
        fallback phase (chain, fresh retries, graceful any-live) stays
        within it.
        """
        limits = getattr(self.settings, "intent_max_fallbacks", {}) or {}
        default = int(getattr(self.settings, "max_model_fallbacks", 10) or 10)
        return max(1, min(int(limits.get(intent, default)), default))

    def _attempt_budget_for(self, intent: str, remaining: float) -> float:
        """Per-intent attempt budget (replaces hardcoded 45.0)."""
        _intent_budgets = getattr(self.settings, "intent_attempt_budget_seconds", {}) or {}
        _default_budget = float(getattr(self.settings, "per_attempt_budget_seconds", 30.0))
        _per_attempt = float(_intent_budgets.get(intent, _default_budget))
        return max(1.0, min(remaining, _per_attempt))

    def _ttft_budget_for(self, model: str, *, is_last: bool) -> float:
        """Uniform TTFT budget: adaptive + speculative fail-fast on every phase.

        Previously the fresh/graceful stream phases used a flat 15s while the
        primary loop used the adaptive hedge — tail latency then depended on
        which phase served the request.
        """
        ttft = float(getattr(self.settings, "stream_ttft_timeout_seconds", 12.0) or 12.0)
        health = getattr(self.registry, "health", None)
        h = health._by_model.get(model) if health is not None else None
        if h is not None and h.ewma_latency > 0:
            ttft = min(ttft, max(3.0, h.ewma_latency * 2.0 + 3.0))
        if getattr(self.settings, "enable_ttft_hedging", True) and not is_last:
            hedge_factor = float(getattr(self.settings, "ttft_hedge_factor", 1.8))
            base_hedge = (h.ewma_latency if h and h.ewma_latency > 0 else 2.0) * hedge_factor
            return min(ttft, max(3.5, base_hedge))
        return ttft

    def _hedge_eligible(self, *, remaining: float, hedges_used: int) -> bool:
        """P2-1: true parallel hedging is opt-in and strictly budgeted
        (per-request cap + enough deadline left for the hedge to matter)."""
        if not getattr(self.settings, "enable_parallel_hedge", False):
            return False
        if hedges_used >= max(
            1, int(getattr(self.settings, "parallel_hedge_max_per_request", 1) or 1)
        ):
            return False
        delay = float(getattr(self.settings, "parallel_hedge_delay_seconds", 2.5) or 2.5)
        return remaining > delay + 5.0

    async def _hedged_first_chunk(
        self,
        *,
        primary_iter: AsyncIterator[bytes],
        primary_model: str,
        primary_pid: str | None,
        primary_key: Any,
        secondary_model: str,
        secondary_pid: str | None,
        open_secondary: Any,
        hedge_delay: float,
        race_timeout: float,
        decision: RouteDecision,
    ) -> tuple[bytes, AsyncIterator[bytes], dict[str, Any]]:
        """P2-1: race primary first-chunk (hedge_delay head start) vs secondary.

        Opens the secondary stream only when the primary is slow, then races
        both first chunks. Returns (first_chunk, full_iter, winner) where
        winner = {"side": "primary"|"secondary", ...}. The loser is closed
        and its TTFT recorded. Raises TimeoutError when neither delivers in
        time (caller takes the TTFT-stall path) and StopAsyncIteration when
        the winning stream is empty (caller takes the empty-stream path).

        Never cancels-then-reuses an iterator: cancelling a waiter athrows
        into the underlying async generator, which would corrupt a stream we
        still intend to serve. Losers are only cancelled once abandoned.
        """
        import asyncio as _aio_h

        def _done_result(task: Any) -> tuple[bool, bytes | None, BaseException | None]:
            """(finished_ok, chunk, error) for a completed anext task."""
            exc = task.exception()
            if exc is None:
                return True, task.result(), None
            if isinstance(exc, StopAsyncIteration):
                return True, b"", None
            return True, None, exc

        primary_task = _aio_h.ensure_future(anext(primary_iter))
        done, _ = await _aio_h.wait({primary_task}, timeout=max(0.05, hedge_delay))
        if primary_task in done:
            ok, chunk, err = _done_result(primary_task)
            if err is not None:
                raise err
            if chunk == b"":
                raise StopAsyncIteration
            return chunk, primary_iter, {"side": "primary"}

        # Primary is slow — open the hedge. The primary waiter keeps running.
        try:
            sec = await open_secondary()
        except Exception:
            # Hedge failed to open — wait out the primary alone.
            done, _ = await _aio_h.wait({primary_task}, timeout=max(1.0, race_timeout))
            if primary_task in done:
                ok, chunk, err = _done_result(primary_task)
                if err is not None:
                    raise err
                if chunk == b"":
                    raise StopAsyncIteration
                return chunk, primary_iter, {"side": "primary"}
            primary_task.cancel()
            with suppress(BaseException):
                await primary_task
            raise TimeoutError("hedged primary TTFT expired")
        self.stats.hedge_attempts += 1
        if sec is None:
            done, _ = await _aio_h.wait({primary_task}, timeout=max(1.0, race_timeout))
            if primary_task in done:
                ok, chunk, err = _done_result(primary_task)
                if err is not None:
                    raise err
                if chunk == b"":
                    raise StopAsyncIteration
                return chunk, primary_iter, {"side": "primary"}
            primary_task.cancel()
            with suppress(BaseException):
                await primary_task
            raise TimeoutError("hedged primary TTFT expired")

        s_status, s_iter, s_headers, s_key = sec
        s_ct = (s_headers.get("content-type") or s_headers.get("Content-Type") or "").lower()
        # H(a): only an SSE secondary can serve the hedge — a 200 JSON body
        # would relay raw JSON bytes to an SSE client. Close it without
        # recording a failure (the model is fine; it just isn't streaming).
        if 200 <= s_status < 300 and "application/json" in s_ct and "text/event-stream" not in s_ct:
            with suppress(Exception):
                aclose = getattr(s_iter, "aclose", None)
                if aclose is not None:
                    await aclose()
            done, _ = await _aio_h.wait({primary_task}, timeout=max(1.0, race_timeout))
            if primary_task in done:
                ok, chunk, err = _done_result(primary_task)
                if err is not None:
                    raise err
                if chunk == b"":
                    raise StopAsyncIteration
                return chunk, primary_iter, {"side": "primary"}
            primary_task.cancel()
            with suppress(BaseException):
                await primary_task
            raise TimeoutError("hedged primary TTFT expired")
        if not (200 <= s_status < 300):
            with suppress(Exception):
                aclose = getattr(s_iter, "aclose", None)
                if aclose is not None:
                    await aclose()
            self._record_circuit(secondary_pid, model=secondary_model, status=s_status)
            self._record_outcome(
                decision, secondary_model, getattr(s_key, "key_id", None),
                success=False, status_code=s_status, intent=decision.intent.value,
            )
            done, _ = await _aio_h.wait({primary_task}, timeout=max(1.0, race_timeout))
            if primary_task in done:
                ok, chunk, err = _done_result(primary_task)
                if err is not None:
                    raise err
                if chunk == b"":
                    raise StopAsyncIteration
                return chunk, primary_iter, {"side": "primary"}
            primary_task.cancel()
            with suppress(BaseException):
                await primary_task
            raise TimeoutError("hedged primary TTFT expired")

        # Race both first chunks.
        sec_task = _aio_h.ensure_future(anext(s_iter))
        done, pending = await _aio_h.wait(
            {primary_task, sec_task},
            timeout=max(1.0, race_timeout),
            return_when=_aio_h.FIRST_COMPLETED,
        )
        if sec_task in done and primary_task not in done:
            sec_exc = sec_task.exception()
            if sec_exc is None:
                # Secondary wins — abandon the slow primary.
                primary_task.cancel()
                with suppress(BaseException):
                    await primary_task
                with suppress(Exception):
                    aclose = getattr(primary_iter, "aclose", None)
                    if aclose is not None:
                        await aclose()
                self._circuit_fail(primary_pid, is_transport=False, model_id=primary_model)
                self._record_outcome(
                    decision, primary_model, getattr(primary_key, "key_id", None),
                    success=False, status_code=504, intent=decision.intent.value,
                )
                self.stats.hedge_wins += 1
                return sec_task.result(), s_iter, {
                    "side": "secondary", "model": secondary_model, "pid": secondary_pid,
                    "key": s_key, "headers": s_headers, "status": s_status,
                }
            # Empty/errored secondary — close it, keep waiting on the primary.
            # A real secondary error is still worth recording (model-tier) so
            # health learns even though the primary decides the request.
            if sec_exc is not None and not isinstance(sec_exc, StopAsyncIteration):
                self._record_circuit(secondary_pid, model=secondary_model, status=502)
                self._record_outcome(
                    decision, secondary_model, getattr(s_key, "key_id", None),
                    success=False, status_code=502, intent=decision.intent.value,
                )
            with suppress(Exception):
                aclose = getattr(s_iter, "aclose", None)
                if aclose is not None:
                    await aclose()
            done, _ = await _aio_h.wait({primary_task}, timeout=max(1.0, race_timeout))
            if primary_task in done:
                ok, chunk, err = _done_result(primary_task)
                if err is not None:
                    raise err
                if chunk == b"":
                    raise StopAsyncIteration
                return chunk, primary_iter, {"side": "primary"}
            primary_task.cancel()
            with suppress(BaseException):
                await primary_task
            raise TimeoutError("hedged primary TTFT expired")
        # Primary finished first (or nothing did) — retire the hedge.
        sec_task.cancel()
        with suppress(BaseException):
            await sec_task
        with suppress(Exception):
            aclose = getattr(s_iter, "aclose", None)
            if aclose is not None:
                await aclose()
        if primary_task in done:
            ok, chunk, err = _done_result(primary_task)
            if err is not None:
                raise err
            if chunk == b"":
                raise StopAsyncIteration
            return chunk, primary_iter, {"side": "primary"}
        for t in pending:
            t.cancel()
            with suppress(BaseException):
                await t
        raise TimeoutError("hedged TTFT expired")
    def _is_auto_decision(self, decision: RouteDecision) -> bool:
        """True when the client asked for auto routing (potato/auto etc.).

        Custom ladders (potato/coding with an admin-defined chain) are NOT
        auto decisions — the admin picked specific models, so we must not
        widen the chain with the live pool.  Detected via rule_id prefix
        ``custom_ladder:`` (set by ModelSelector.resolve for admin ladders).
        """
        if str(decision.rule_id or "").startswith("custom_ladder:"):
            return False
        mode = str(decision.mode or "")
        if mode in {"auto", "unknown_alias_as_auto", "alias"}:
            return True
        if getattr(decision, "auto_tier", None):
            return True
        req = str(decision.requested_model or "").lower()
        if not req or req in {"auto", ""}:
            return True
        try:
            from potato.routing.auto_router import is_auto_router_id

            return is_auto_router_id(req)
        except Exception:
            return False

    def _heal_empty_chain(
        self,
        decision: RouteDecision,
        *,
        max_n: int,
        disabled: set[str],
        had_tools: bool,
    ) -> list[str]:
        """Rebuild a non-empty intent-aware chain when the primary path is empty."""
        from potato.routing.auto_router import (
            build_intent_aware_pool,
            filter_chain,
        )

        intent = decision.intent.value
        variant = getattr(decision, "variant", None) or "default"
        pool = build_intent_aware_pool(
            self.registry,
            primary_intent=intent,
            variant=variant,
            max_n=max(max_n * 2, 16),
            include_related=True,
        )
        available = [
            m
            for m in pool
            if self._provider_available(m)
            and (self.registry.resolve_live_id(m, include_disabled=True) or m) not in disabled
        ]
        if had_tools and hasattr(self.registry, "ladder"):
            caps = getattr(self.registry.ladder, "capabilities", {})
            tool_ok = [
                m for m in available if (caps.get(m) or {}).get("supports_tools") is not False
            ]
            if tool_ok:
                available = tool_ok
        allowed = list(getattr(decision, "allowed_models", None) or [])
        free_only = str(getattr(decision, "auto_tier", "") or "").lower() == "free"
        if allowed or free_only:
            available = filter_chain(available, allowed_models=allowed or None, free_only=free_only)
        return available

    def _chain(self, decision: RouteDecision, *, had_tools: bool = False) -> list[str]:
        max_n = self._max_n_for_intent(decision.intent.value)
        is_auto = self._is_auto_decision(decision)

        raw = list(decision.chain)
        # For auto: widen with related-intent models before execution so
        # exhaustion of the primary ladder still processes the request.
        # Intent-escalation: when auto, append models from higher-capability
        # intents after the primary chain so a misclassified or underestimated
        # intent can still be served.  Expansion is limited to models already
        # in the chain or live_ids so tests / constrained pools are not polluted
        # with random catalog entries.
        if is_auto:
            try:
                from potato.routing.auto_router import (
                    build_intent_aware_pool,
                    intent_expansion_order,
                )

                live_set = getattr(self.registry, "live_ids", None) or set()
                chain_set = {m.lower() for m in raw}
                # When live_ids is populated, expansion adds models from the
                # catalog that are live.  When live_ids is empty (unit tests,
                # minimal setups), skip expansion to preserve the test chain.
                if live_set:
                    expanded = build_intent_aware_pool(
                        self.registry,
                        primary_intent=decision.intent.value,
                        variant=getattr(decision, "variant", None) or "default",
                        max_n=max(max_n * 2, 20),
                        include_related=True,
                    )
                    if expanded:
                        seen = {m.lower() for m in raw}
                        for m in expanded:
                            ml = m.lower()
                            if ml not in seen and (ml in live_set or ml in chain_set):
                                raw.append(m)
                                seen.add(ml)
                    # Intent-escalation tail: append models from higher-capability
                    # intents that are already in the live pool, so a chat_fast
                    # request that turns out to be agentic still reaches coding.
                    seen = {m.lower() for m in raw}
                    escalation_intents = [
                        i
                        for i in intent_expansion_order(decision.intent.value)
                        if i != decision.intent.value
                    ]
                    for esc_intent in escalation_intents:
                        try:
                            esc_chain = self.registry.chain_for_intent(
                                esc_intent,
                                variant=getattr(decision, "variant", None) or "default",
                            )
                            for m in esc_chain or []:
                                ml = m.lower()
                                if ml not in seen and ml in live_set:
                                    raw.append(m)
                                    seen.add(ml)
                        except Exception:
                            pass
            except Exception:
                logger.debug("auto chain expansion failed", exc_info=True)

        # Never execute admin-disabled models (covers passthrough / emergency paths)
        disabled = getattr(self.registry, "disabled_models", None) or set()
        if disabled:
            raw = [
                m
                for m in raw
                if (self.registry.resolve_live_id(m, include_disabled=True) or m) not in disabled
            ]
        # Pre-filter: skip models confirmed to not support tools when tools are present
        if had_tools and hasattr(self.registry, "ladder"):
            caps = getattr(self.registry.ladder, "capabilities", {})
            filtered = []
            for m in raw:
                cap = caps.get(m) or {}
                if cap.get("supports_tools") is False:
                    logger.debug("pre-filter: skipping %s (tools not supported)", m)
                    continue
                filtered.append(m)
            # Keep at least one candidate even if all claim no tools (capability may be stale)
            if filtered:
                raw = filtered
            elif is_auto and raw:
                logger.info(
                    "tools pre-filter emptied chain; keeping candidates for auto intent=%s",
                    decision.intent.value,
                )
        # Drop models whose provider has no active keys/runtime (production safety)
        available = [m for m in raw if self._provider_available(m)]
        # Resilience: drop models whose provider circuit is hard-open so the
        # next-best provider actually gets picked instead of burning attempts
        # on a dead provider. Keep at least one candidate when every provider
        # is down — client_for_model force-allows the last resort then.
        if self.hub is not None and len(available) > 1:
            cb = getattr(self.hub, "circuit_breaker", None)
            if cb is not None:
                circuit_open = [m for m in available if cb.blocked(self._provider_id_for(m))]
                if circuit_open and len(circuit_open) < len(available):
                    dropped = set(circuit_open)
                    available = [m for m in available if m not in dropped]
                    logger.info(
                        "circuit filter: dropped %s model(s) on open provider circuit "
                        "(intent=%s)",
                        len(circuit_open),
                        decision.intent.value,
                    )
        # D3/P0-3c: drop models on breaker's model-tier cooldown (keep at
        # least one candidate — the executor force-allows the last resort).
        if self.hub is not None and len(available) > 1:
            cooled = [m for m in available if self._model_on_cooldown(m)]
            if cooled and len(cooled) < len(available):
                available = [m for m in available if not self._model_on_cooldown(m)]
                logger.info(
                    "model-cooldown filter: dropped %s model(s) (intent=%s)",
                    len(cooled),
                    decision.intent.value,
                )
        if not available:
            # Self-heal: intent-aware multi-ladder rebuild, then emergency
            try:
                available = self._heal_empty_chain(
                    decision,
                    max_n=max_n,
                    disabled=disabled,
                    had_tools=had_tools,
                )
                if available:
                    logger.warning(
                        "empty chain healed with %s intent-aware models (intent=%s)",
                        len(available),
                        decision.intent.value,
                    )
            except Exception:
                logger.exception("intent-aware chain heal failed")
        if not available and raw:
            logger.warning("all %s chain models have unavailable providers", len(raw))
        # Continuous optimizer: intelligence × speed × health (every request)
        intent = decision.intent.value
        variant = getattr(decision, "variant", None) or "default"
        if variant == "default":
            req = str(decision.requested_model or "").lower()
            tier = str(getattr(decision, "auto_tier", "") or "").lower()
            if "cheap" in req or tier in ("efficient", "free") or "efficient" in req:
                variant = "cheap"
            elif "fast" in req or tier == "fast":
                variant = "fast"
        pinned = getattr(decision, "pinned_head", None) or getattr(decision, "sticky_model", None)
        # Drop pin if it was admin-disabled
        if pinned and disabled:
            pin_live = self.registry.resolve_live_id(pinned, include_disabled=True) or pinned
            if pin_live in disabled:
                pinned = None
        # Auto intent-strict: demote pin outside the available intent pool
        if is_auto and pinned and available and pinned not in available:
            pinned = None
        # Re-rank, but keep pinned head first unless unhealthy (F-08)
        # Skip re-optimization if no filtering changed the chain (selector already optimized)
        if available:
            from potato.routing.optimizer import optimize_chain

            needs_optimize = available != raw or pinned
            if needs_optimize and pinned and pinned in available:
                tail = [m for m in available if m != pinned]
                tail = optimize_chain(
                    tail,
                    self.registry,
                    intent=intent,
                    variant=variant,
                    max_n=None,
                )
                unhealthy = hasattr(self.registry, "health") and self.registry.health.is_unhealthy(
                    pinned
                )
                if unhealthy:
                    logger.info("pin_demoted model=%s reason=unhealthy", pinned)
                    available = optimize_chain(
                        [pinned] + tail,
                        self.registry,
                        intent=intent,
                        variant=variant,
                        max_n=None,
                    )
                else:
                    available = [pinned] + tail
            elif needs_optimize:
                available = optimize_chain(
                    available,
                    self.registry,
                    intent=intent,
                    variant=variant,
                    max_n=None,
                )
        # LinUCB contextual re-rank: boost models that learned high reward for
        # this request's feature vector.  Applied after quality×speed×health
        # optimization so the bandit only re-orders within the already-vetted
        # chain.  Pinned head stays first (F-08).  ponytail: no-op until the
        # bandit has ≥1 sample per model — cold start preserves base order.
        rl_x = getattr(decision, "feature_vector", None)
        if self.rl_engine is not None and rl_x and len(rl_x) == 12 and available and len(available) > 1:
            try:
                if pinned and pinned in available:
                    tail = [m for m in available if m != pinned]
                    scored: list[tuple[float, str]] = []
                    for idx, mid in enumerate(tail):
                        rl_score, _, _ = self.rl_engine.score(mid, rl_x)
                        boost = max(0.5, min(2.0, 1.0 + rl_score))
                        position_weight = 1.0 / (1.0 + 0.01 * idx)
                        scored.append((boost * position_weight, mid))
                    scored.sort(key=lambda t: t[0], reverse=True)
                    available = [pinned] + [m for _, m in scored]
                else:
                    scored = []
                    for idx, mid in enumerate(available):
                        rl_score, _, _ = self.rl_engine.score(mid, rl_x)
                        boost = max(0.5, min(2.0, 1.0 + rl_score))
                        position_weight = 1.0 / (1.0 + 0.01 * idx)
                        scored.append((boost * position_weight, mid))
                    scored.sort(key=lambda t: t[0], reverse=True)
                    available = [m for _, m in scored]
            except Exception:
                logger.debug("RL chain re-rank failed", exc_info=True)
        # Fail-fast: skip cooling models for TTFT (keep 1 cold last-resort)
        # Preserve pinned head if healthy. Auto keeps more cold models as safety net.
        if available and hasattr(self.registry, "health"):
            hot = [m for m in available if not self.registry.health.is_unhealthy(m)]
            cold = [m for m in available if self.registry.health.is_unhealthy(m)]
            if pinned and pinned in hot:
                hot = [pinned] + [m for m in hot if m != pinned]
            # Auto keeps more cold models: escalation tail from related intents
            # may be cooling but still the only path to intent fulfillment.
            cold_keep = min(len(cold), 6 if is_auto else 1)
            available = hot + cold[:cold_keep]
        # Quality floor: drop models below min_quality_ratio × top model quality.
        # For auto: never empty the chain — prefer a lower-quality model over 503.
        # Auto floor is deliberately low (0.35) so escalation-tail models are
        # not pruned before they get a chance to serve.
        if available and len(available) > 1:
            min_ratio = float(getattr(self.settings, "min_quality_ratio", 0.6) or 0.6)
            # Auto softens floor so intent can still be served under pool pressure
            if is_auto:
                min_ratio = min(min_ratio, 0.35)
            ladder = getattr(self.registry, "ladder", None)
            if ladder is not None:
                snap = getattr(ladder, "_ladders", {}).get((intent, variant))
                if snap is not None and getattr(snap, "scores", None):
                    scores = snap.scores
                    top_score = max(scores.get(m, 0.0) for m in available) or 1.0
                    floor = top_score * min_ratio
                    filtered = [m for m in available if scores.get(m, 0.0) >= floor]
                    if filtered:
                        available = filtered
                    elif is_auto:
                        # Keep original ranking — serving beats silence
                        logger.info(
                            "quality floor would empty auto chain intent=%s; keeping all",
                            intent,
                        )
        # Drop models whose known context cannot fit the estimate (T13)
        est = getattr(decision, "estimated_tokens", None)
        if est and available:
            fit: list[str] = []
            unknown: list[str] = []
            overflow: list[str] = []
            for m in available:
                ctx_len = self.registry.context_length_for(m)
                if ctx_len is None:
                    unknown.append(m)
                elif ctx_len >= est:
                    fit.append(m)
                else:
                    overflow.append(m)
            if fit or unknown:
                # Auto: keep overflow models at the very tail (provider may still accept)
                available = fit + unknown + (overflow if is_auto else [])
            elif is_auto and overflow:
                # Only overflow known — still try rather than 503
                available = overflow
        # Hard routing constraints (allowed_models / free-only) must hold for
        # every model that reaches execution — auto expansion, escalation
        # tails, and healing can add models outside the caller's allowlist.
        allowed = list(getattr(decision, "allowed_models", None) or [])
        free_only = str(getattr(decision, "auto_tier", "") or "").lower() == "free"
        if allowed or free_only:
            from potato.routing.auto_router import filter_chain

            available = filter_chain(available, allowed_models=allowed or None, free_only=free_only)
        # Anti-celebrity exploration slot: on auto decisions, promote one
        # under-sampled live model (highest UCB) to second place so the bandit
        # actually samples challengers instead of locking in the ladder
        # leader. Quality-gated so exploration never serves a model far
        # below the proven head; constraint filters already applied above.
        if (
            is_auto
            and len(available) > 1
            and getattr(self.settings, "rl_exploration_enabled", True)
        ):
            try:
                challenger = self._best_exploration_candidate(
                    available,
                    intent=intent,
                    allowed_models=allowed,
                    free_only=free_only,
                )
                if challenger:
                    if challenger in available[:2]:
                        challenger = None  # already leading — no slot needed
                    else:
                        # Move (not duplicate) the challenger to second place.
                        available = [m for m in available if m != challenger]
                        available.insert(1, challenger)
            except Exception:
                logger.debug("exploration slot selection failed", exc_info=True)
        chain = self._enforce_provider_diversity(available)[: max(1, max_n)]
        # Final auto guarantee
        if not chain and is_auto:
            try:
                chain = self._heal_empty_chain(
                    decision,
                    max_n=max_n,
                    disabled=disabled,
                    had_tools=had_tools,
                )[: max(1, max_n)]
            except Exception:
                logger.exception("final auto chain heal failed")
        return chain

    def routing_headers(
        self,
        decision: RouteDecision,
        *,
        model: str,
        key_id: str | None,
        fallback_index: int,
        provider_id: str | None = None,
    ) -> dict[str, str]:
        h = {
            "X-Potato-Model": model,
            "X-Potato-Intent": decision.intent.value,
            "X-Potato-Route-Mode": decision.mode,
            "X-Potato-Fallback-Index": str(fallback_index),
            "X-Potato-Rule-Id": decision.rule_id,
        }
        if key_id:
            h["X-Potato-Key-Id"] = key_id
        if decision.requested_model:
            h["X-Potato-Requested-Model"] = str(decision.requested_model)
        if getattr(decision, "auto_tier", None):
            h["X-Potato-Auto-Tier"] = str(decision.auto_tier)
        if getattr(decision, "sticky_model", None):
            h["X-Potato-Sticky-Model"] = str(decision.sticky_model)
        ctx_len = self.registry.context_length_for(model)
        if ctx_len is not None:
            h["X-Potato-Context-Length"] = str(ctx_len)
        pid = provider_id or self._provider_id_for(model)
        if pid:
            h["X-Potato-Provider"] = pid
        return h

    async def execute_json(
        self,
        path: str,
        body: dict[str, Any],
        decision: RouteDecision,
        *,
        preferred_key_id: str | None = None,
        forward_headers: dict[str, str] | None = None,
        fallback_on_pool_exhaust: bool | None = None,
    ) -> UpstreamResult:
        had_tools = bool(
            (body.get("tools") or body.get("functions"))
            or body.get("tool_choice") not in (None, "none", "None")
        )
        chain = self._chain(decision, had_tools=had_tools)
        if not chain:
            return UpstreamResult(
                status_code=503,
                body={
                    "error": {
                        "message": "No models available in routing chain.",
                        "type": "server_error",
                        "code": "potato_catalog_empty",
                    }
                },
                headers={},
                key=None,
                model="",
                fallback_index=0,
                decision=decision,
            )

        advance_on_pool = (
            self.settings.fallback_on_pool_exhaust
            if fallback_on_pool_exhaust is None
            else fallback_on_pool_exhaust
        )
        last: UpstreamResult | None = None

        import httpx

        from potato.compat import openai_error

        deadline = self._make_deadline(decision.intent.value)
        chain_budget = max(1, self._max_n_for_intent(decision.intent.value))
        # D1: recovery (last-resort + graceful) must not be budget-starved by
        # the primary chain. Reserve attempts for recovery phases; total stays
        # bounded by chain_budget + recovery_budget and by the deadline.
        # cap == 1 -> reserve 0 (universal ceiling semantics preserved).
        recovery_budget = max(0, min(3, chain_budget - 1))
        max_attempts = chain_budget + recovery_budget
        attempts = 0

        for idx, model in enumerate(chain):
            remaining = deadline - time.monotonic()
            if remaining < 1.0 and idx > 0:
                return UpstreamResult(
                    status_code=504,
                    body=openai_error(
                        "Request deadline exceeded before trying remaining models.",
                        code="request_deadline_exceeded",
                        type_="server_error",
                    ),
                    headers={},
                    key=None,
                    model=last.model if last else model,
                    fallback_index=idx,
                    decision=decision,
                )
            try:
                client, upstream_mid = self._client_for(model)
            except RuntimeError as exc:
                if idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    logger.info("client_for_model failed on %s: %s; advancing", model, exc)
                    continue
                # Last model's provider unavailable — do NOT fail cold; fall
                # through to the last-resort force-allow + fresh-model retry.
                pid_fail = self._provider_id_for(model)
                self._record_outcome(
                    decision,
                    model,
                    None,
                    success=False,
                    status_code=503,
                    unavailable=True,
                    intent=decision.intent.value,
                )
                last = UpstreamResult(
                    status_code=503,
                    body={
                        "error": {
                            "message": str(exc),
                            "type": "server_error",
                            "code": "potato_provider_unavailable",
                        }
                    },
                    headers={},
                    key=None,
                    model=model,
                    fallback_index=idx,
                    decision=decision,
                    provider_id=pid_fail,
                )
                break
            attempt_body = {**body, "model": upstream_mid}
            # Per-model reasoning_effort normalization (resilience: a reasoning
            # head failing over to a non-reasoning model strips the field).
            from potato.compat import normalize_reasoning_effort as _nre

            attempt_body = _nre(
                attempt_body,
                routed_model=model,
                registry=self.registry,
                default_effort=getattr(self.settings, "default_reasoning_effort", ""),
            )
            t_attempt = time.perf_counter()
            pid = self._provider_id_for(model)
            # Apply per-model recommendations (temperature, max_tokens)
            if hasattr(self.registry, "ladder"):
                rec = self.registry.ladder.model_recommendations(model)
                if rec:
                    # Cap max_tokens to model limit if client didn't set it
                    max_limit = rec.get("max_tokens_limit")
                    if max_limit and not attempt_body.get("max_tokens"):
                        attempt_body["max_tokens"] = max_limit
                    elif max_limit and attempt_body.get("max_tokens"):
                        attempt_body["max_tokens"] = min(attempt_body["max_tokens"], max_limit)
            # Log large payloads for debugging agentic loop context overflow
            import sys

            body_size = sys.getsizeof(str(attempt_body))
            if body_size > 100_000:  # >100KB
                logger.info(
                    "large request body: model=%s size=%dKB messages=%d",
                    model,
                    body_size // 1024,
                    len(attempt_body.get("messages") or []),
                )
            try:
                import asyncio as _aio

                attempt_budget = self._attempt_budget_for(decision.intent.value, remaining)
                status, resp_body, headers, key = await _aio.wait_for(
                    client.request_json(
                        "POST",
                        path,
                        json_body=attempt_body,
                        forward_headers=forward_headers,
                        preferred_key_id=preferred_key_id,
                        max_retries=2,
                        # H(b): bound the acquire wait so pool-exhaustion
                        # surfaces as capacity, not a transport stall.
                        acquire_timeout=min(2.0, attempt_budget),
                    ),
                    timeout=attempt_budget,
                )
            except TimeoutError:
                attempts += 1
                self._circuit_fail(pid, is_transport=False, model_id=model)
                self._record_outcome(
                    decision,
                    model,
                    getattr(key, "key_id", None) if "key" in locals() else None,
                    success=False,
                    status_code=504,
                    intent=decision.intent.value,
                    latency=(time.perf_counter() - t_attempt),
                )
                self._emit_span(
                    self._make_upstream_span(
                        model=model,
                        t0=t_attempt,
                        status=504,
                        success=False,
                        error_message="attempt_deadline_exceeded",
                        span_type="fallback_advance" if idx < len(chain) - 1 else "upstream",
                    )
                )
                if idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(2.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                    logger.info("json attempt deadline on %s; falling back", model)
                    continue
                # Last model timeout — do NOT return cold; fall through to
                # last-resort force-allow + fresh-model retry.
                last = UpstreamResult(
                    status_code=504,
                    body=openai_error(
                        "Upstream attempt exceeded request deadline.",
                        code="request_deadline_exceeded",
                        type_="server_error",
                    ),
                    headers={},
                    key=None,
                    model=model,
                    fallback_index=idx,
                    decision=decision,
                    provider_id=pid,
                )
                break
            except (RuntimeError, httpx.HTTPError, OSError) as exc:
                attempts += 1
                # D3: capacity (pool exhaustion) cools the model tier only.
                if isinstance(exc, KeyPoolExhausted):
                    if self.hub is not None:
                        cb = getattr(self.hub, "circuit_breaker", None)
                        if cb is not None:
                            cb.fail(pid or "", is_transport=False, model_id=model)
                else:
                    self._circuit_fail(pid, model_id=model)
                self._record_outcome(
                    decision,
                    model,
                    getattr(key, "key_id", None) if "key" in locals() else None,
                    success=False,
                    status_code=503,
                    intent=decision.intent.value,
                    latency=(time.perf_counter() - t_attempt),
                )
                self._emit_span(
                    self._make_upstream_span(
                        model=model,
                        t0=t_attempt,
                        status=503,
                        success=False,
                        error_message=str(exc),
                        span_type="fallback_advance"
                        if idx < len(chain) - 1
                        else "upstream",
                    )
                )
                if idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(2.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                    logger.info(
                        "provider/transport error on %s (%s); advancing model",
                        model,
                        exc,
                    )
                    continue
                # Last model error — do NOT return cold; fall through to
                # last-resort force-allow + fresh-model retry.
                last = UpstreamResult(
                    status_code=503,
                    body={
                        "error": {
                            "message": str(exc),
                            "type": "server_error",
                            "code": (
                                "potato_pool_exhausted"
                                if isinstance(exc, KeyPoolExhausted)
                                else "potato_upstream_error"
                            ),
                        }
                    },
                    headers={},
                    key=None,
                    model=model,
                    fallback_index=idx,
                    decision=decision,
                    upstream_ms=(time.perf_counter() - t_attempt) * 1000,
                    provider_id=pid,
                )
                break

            attempts += 1
            key_id = key.key_id if key else None
            unavailable = _is_model_not_found(status, resp_body)
            success = 200 <= status < 300
            if success:
                self._circuit_succeed(pid, model_id=model)
            elif status >= 500:
                self._record_circuit(pid, model=model, status=status)
            had_tools = bool(
                (body.get("tools") or body.get("functions"))
                or body.get("tool_choice") not in (None, "none", "None")
            )
            empty_reply = False
            tool_ok: bool | None = None
            if success:
                empty_reply, tool_ok = _analyze_success_body(
                    resp_body, had_tools=had_tools, path=path
                )
            # Adaptive speed signal: JSON path latency (if measured upstream)
            latency = (time.perf_counter() - t_attempt) * 1000
            tokens = None
            pt = ct = cached = 0
            if success and isinstance(resp_body, dict):
                usage = resp_body.get("usage")
                if isinstance(usage, dict):
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    cached = int(
                        usage.get("cached_tokens")
                        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                        or 0
                    )
                    tokens = pt + ct if (pt or ct) else None
            self._emit_span(
                self._make_upstream_span(
                    model=model,
                    t0=t_attempt,
                    status=status,
                    success=success and not ((had_tools and tool_ok is False) or empty_reply),
                    error_message=None if success else f"upstream_{status}",
                    metadata={
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "cached_tokens": cached,
                        "empty_reply": empty_reply,
                        "tool_ok": tool_ok,
                    },
                    span_type="upstream",
                )
            )
            self._record_outcome(
                decision,
                model,
                key_id,
                success=success,
                latency=latency / 1000.0 if latency else None,
                tokens=tokens,
                status_code=status,
                unavailable=unavailable,
                intent=decision.intent.value,
                empty_reply=empty_reply,
                had_tools=had_tools,
                tool_ok=tool_ok,
            )
            if had_tools and tool_ok is True:
                self.registry.ladder.set_capability(model, supports_tools=True)
            elif had_tools and tool_ok is False and success:
                # Don't mark unsupported on empty once — wait for learning demotion
                pass
            body_l = str(resp_body).lower()
            if (unavailable or status == 400) and "tool" in body_l and "support" in body_l:
                self.registry.ladder.set_capability(model, supports_tools=False)

            if success:
                soft_fail = (had_tools and tool_ok is False) or empty_reply
                if soft_fail and idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    logger.info(
                        "model %s soft-fail (empty=%s tool_ok=%s); falling back",
                        model,
                        empty_reply,
                        tool_ok,
                    )
                    continue
                if isinstance(resp_body, dict):
                    if "model" in resp_body:
                        resp_body = {**resp_body, "model": model}
                    usage = resp_body.get("usage")
                    if isinstance(usage, dict):
                        pt = int(usage.get("prompt_tokens") or 0)
                        ct = int(usage.get("completion_tokens") or 0)
                        cached = int(
                            usage.get("cached_tokens")
                            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                            or 0
                        )
                        self.stats.record_tokens(model, key_id, pt, ct)
                self.stats.record(decision.intent.value, model, advanced=idx > 0)
                return UpstreamResult(
                    status_code=status,
                    body=resp_body,
                    headers=headers,
                    key=key,
                    model=model,
                    fallback_index=idx,
                    decision=decision,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    cached_tokens=cached,
                    upstream_ms=latency,
                    provider_id=pid,
                )

            from potato.compat import wrap_upstream_error

            last = UpstreamResult(
                status_code=status,
                body=wrap_upstream_error(resp_body, status=status),
                headers=headers,
                key=key,
                model=model,
                fallback_index=idx,
                decision=decision,
                upstream_ms=latency,
                provider_id=pid,
            )

            if _is_non_retryable_client_error(status, resp_body):
                self.stats.record(decision.intent.value, model, advanced=False)
                return last

            if _is_retryable_model_error(status, resp_body) and idx < len(chain) - 1:
                # 504 = upstream gateway timeout — already timed out, advance now (no sleep).
                # 429 = rate limited — model/key cooldown already recorded, so
                # advance immediately (sleeping only burns the deadline when
                # other candidates are available).
                # 503 = transient overload — tiny backoff then advance.
                if status == 504:
                    pass  # no backoff — advance immediately
                elif status == 429:
                    pass  # no backoff — advance immediately
                elif status in {500, 502, 503}:
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(1.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                self.stats.fallback_advances += 1
                logger.info(
                    "model %s failed status=%s; falling back (%s/%s)",
                    model,
                    status,
                    idx + 1,
                    len(chain),
                )
                continue

            # 429 after key retries — optionally advance (no sleep: the key
            # cooldown is recorded, and sleeping only burns the deadline).
            if status == 429 and advance_on_pool and idx < len(chain) - 1:
                self.stats.fallback_advances += 1
                continue

            self.stats.record(decision.intent.value, model, advanced=idx > 0)
            break  # don't return — fall through to last-resort + graceful fallback

        assert last is not None
        fresh: list[str] = []
        if last.status_code >= 400:
            # ── Last-resort: retry fresh models from the widest pool ──
            # D4: scoped recovery — never wipe every model cooldown or
            # force-allow every provider (that defeats the breaker during
            # incidents). hub.client_for_model already force-allows when ALL
            # providers are open; single-provider recovery uses probe timers.
            # P1-4: recovery runs until 85% of the deadline; the final slice
            # is reserved for the structured error envelope + logging.
            phases = self._phase_deadlines(deadline, time.monotonic())
            if time.monotonic() < phases["recovery"] and attempts < max_attempts:
                self.stats.recovery_entered += 1
                tried = {m.lower() for m in chain}
                can_recover_wide = self._is_auto_decision(decision) or getattr(
                    self.settings, "allow_graceful_fallback_on_explicit", False
                )
                fresh, _retry_chain = self._recovery_pool(
                    decision, had_tools=had_tools, tried=tried,
                    allow_any_live=can_recover_wide,
                )
                # Relief only for models about to be retried.
                self._clear_model_cooldowns(fresh)
                if fresh:
                    logger.warning(
                        "last-resort: retrying %s intent-aware models (intent=%s)",
                        len(fresh),
                        decision.intent.value,
                    )
                    import asyncio as _aio2

                    from potato.compat import wrap_upstream_error as _wue

                    for idx2, model2 in enumerate(fresh):
                        rem2 = deadline - time.monotonic()
                        if rem2 < 3.0:
                            break
                        if attempts >= max_attempts:
                            break
                        try:
                            client2, upstream_mid2 = self._client_for(model2)
                        except Exception:
                            # D2: skip — never a transport failure.
                            continue
                        attempt_body2 = {**body, "model": upstream_mid2}
                        if hasattr(self.registry, "ladder"):
                            rec = self.registry.ladder.model_recommendations(model2)
                            if rec:
                                ml = rec.get("max_tokens_limit")
                                if ml and not attempt_body2.get("max_tokens"):
                                    attempt_body2["max_tokens"] = ml
                                elif ml and attempt_body2.get("max_tokens"):
                                    attempt_body2["max_tokens"] = min(
                                        attempt_body2["max_tokens"], ml
                                    )
                        pid2 = self._provider_id_for(model2)
                        t2 = time.perf_counter()
                        try:
                            budget2 = self._attempt_budget_for(decision.intent.value, rem2)
                            s2, rb2, hd2, k2 = await _aio2.wait_for(
                                client2.request_json(
                                    "POST",
                                    path,
                                    json_body=attempt_body2,
                                    forward_headers=forward_headers,
                                    preferred_key_id=preferred_key_id,
                                    max_retries=2,
                                    acquire_timeout=min(2.0, budget2),
                                ),
                                timeout=budget2,
                            )
                        except TimeoutError:
                            attempts += 1
                            self._circuit_fail(pid2, model_id=model2)
                            self._record_outcome(
                                decision,
                                model2,
                                None,
                                success=False,
                                status_code=504,
                                intent=decision.intent.value,
                            )
                            continue
                        except (RuntimeError, httpx.HTTPError, OSError) as exc2:
                            attempts += 1
                            # D3: capacity cools the model tier only.
                            if isinstance(exc2, KeyPoolExhausted):
                                if self.hub is not None:
                                    cb = getattr(self.hub, "circuit_breaker", None)
                                    if cb is not None:
                                        cb.fail(pid2 or "", is_transport=False, model_id=model2)
                            else:
                                self._circuit_fail(pid2, model_id=model2)
                            self._record_outcome(
                                decision,
                                model2,
                                None,
                                success=False,
                                status_code=503,
                                intent=decision.intent.value,
                            )
                            continue

                        attempts += 1
                        lat2 = (time.perf_counter() - t2) * 1000
                        if 200 <= s2 < 300:
                            self._circuit_succeed(pid2, model_id=model2)
                            empty2, tool_ok2 = _analyze_success_body(
                                rb2, had_tools=had_tools, path=path
                            )
                            if empty2 or (had_tools and tool_ok2 is False):
                                self._record_outcome(
                                    decision,
                                    model2,
                                    k2.key_id if k2 else None,
                                    success=False,
                                    status_code=s2,
                                    intent=decision.intent.value,
                                    empty_reply=empty2,
                                    had_tools=had_tools,
                                    tool_ok=tool_ok2,
                                )
                                last = UpstreamResult(
                                    status_code=s2,
                                    body=rb2,
                                    headers=hd2,
                                    key=k2,
                                    model=model2,
                                    fallback_index=len(chain) + idx2,
                                    decision=decision,
                                    upstream_ms=lat2,
                                    provider_id=pid2,
                                )
                                continue
                            pt2 = ct2 = cached2 = 0
                            if isinstance(rb2, dict):
                                usage2 = rb2.get("usage")
                                if isinstance(usage2, dict):
                                    pt2 = int(usage2.get("prompt_tokens") or 0)
                                    ct2 = int(usage2.get("completion_tokens") or 0)
                                    cached2 = int(
                                        usage2.get("cached_tokens")
                                        or (usage2.get("prompt_tokens_details") or {}).get(
                                            "cached_tokens", 0
                                        )
                                        or 0
                                    )
                            self._record_outcome(
                                decision,
                                model2,
                                k2.key_id if k2 else None,
                                success=True,
                                latency=lat2 / 1000,
                                status_code=s2,
                                intent=decision.intent.value,
                                empty_reply=empty2,
                                had_tools=had_tools,
                                tool_ok=tool_ok2,
                            )
                            self.stats.record(decision.intent.value, model2, advanced=True)
                            self.stats.record_tokens(model2, k2.key_id if k2 else None, pt2, ct2)
                            if isinstance(rb2, dict) and "model" in rb2:
                                rb2 = {**rb2, "model": model2}
                            return UpstreamResult(
                                status_code=s2,
                                body=rb2,
                                headers=hd2,
                                key=k2,
                                model=model2,
                                fallback_index=len(chain) + idx2,
                                decision=decision,
                                prompt_tokens=pt2,
                                completion_tokens=ct2,
                                cached_tokens=cached2,
                                upstream_ms=lat2,
                                provider_id=pid2,
                            )
                        # Failed — record and try next fresh model
                        self._record_circuit(pid2, model=model2, status=s2)
                        self._record_outcome(
                            decision,
                            model2,
                            k2.key_id if k2 else None,
                            success=False,
                            status_code=s2,
                            intent=decision.intent.value,
                        )
                        last = UpstreamResult(
                            status_code=s2,
                            body=_wue(rb2, status=s2),
                            headers=hd2,
                            key=k2,
                            model=model2,
                            fallback_index=len(chain) + idx2,
                            decision=decision,
                            upstream_ms=lat2,
                            provider_id=pid2,
                        )

            # ── Graceful fallback: try ANY live model from ANY provider ──
            # When the intent-aware chain and fresh retries are exhausted,
            # cast the widest net: every live model whose provider has a
            # runtime. Better to serve with a "wrong-intent" model than 503.
            can_graceful = self._is_auto_decision(decision) or getattr(
                self.settings, "allow_graceful_fallback_on_explicit", False
            )
            if last.status_code >= 400 and can_graceful:
                phases = self._phase_deadlines(deadline, time.monotonic())
                if time.monotonic() < phases["recovery"] and attempts < max_attempts:
                    try:
                        any_live = self._any_available_live_models(
                            had_tools=had_tools,
                            intent=decision.intent.value,
                            allowed_models=list(getattr(decision, "allowed_models", None) or []),
                            free_only=(
                                str(getattr(decision, "auto_tier", "") or "").lower() == "free"
                            ),
                        )
                        already = {m.lower() for m in chain} | {m.lower() for m in fresh}
                        untried_any = [m for m in any_live if m.lower() not in already]
                        if untried_any:
                            logger.warning(
                                "graceful fallback: trying %s any-provider "
                                "live models (intent=%s, chain exhausted)",
                                len(untried_any),
                                decision.intent.value,
                            )
                            last = await self._try_models(
                                untried_any,
                                body=body,
                                path=path,
                                decision=decision,
                                deadline=deadline,
                                forward_headers=forward_headers,
                                preferred_key_id=preferred_key_id,
                                had_tools=had_tools,
                                chain_len=len(chain),
                                start_idx=len(chain),
                                last=last,
                                max_attempts=max(0, max_attempts - attempts),
                            )
                    except Exception:
                        logger.exception("graceful fallback failed")

            # Only build the terminal envelope if the graceful fallback didn't
            # produce a success (last was overwritten by _try_models on success).
            # D5d: never surface raw provider text as the gateway message, and
            # always include Retry-After so clients can back off.
            if last.status_code >= 400:
                last = UpstreamResult(
                    status_code=503,
                    body=openai_error(
                        "All models in routing chain failed after recovery.",
                        code="all_providers_failed",
                        type_="server_error",
                        metadata={
                            "retry_after": 3,
                            "last_status": last.status_code,
                            "last_provider": last.provider_id,
                            "upstream_detail": str(last.body)[:300],
                        },
                    ),
                    headers={**last.headers, "Retry-After": "3"},
                    key=last.key,
                    model=last.model,
                    fallback_index=last.fallback_index,
                    decision=decision,
                )
                self.stats.chain_exhausted += 1
        self.stats.record(decision.intent.value, last.model, advanced=True)
        return last

    async def execute_stream(
        self,
        path: str,
        body: dict[str, Any],
        decision: RouteDecision,
        *,
        preferred_key_id: str | None = None,
        forward_headers: dict[str, str] | None = None,
    ) -> StreamResult:
        """
        Try models until a stream opens successfully. Never switch mid-stream.
        """
        had_tools = bool(
            (body.get("tools") or body.get("functions"))
            or body.get("tool_choice") not in (None, "none", "None")
        )
        chain = self._chain(decision, had_tools=had_tools)
        from potato.compat import (
            frame_sse_error,
            json_body_to_sse,
            openai_error,
            wrap_upstream_error,
        )

        if not chain:
            payload = frame_sse_error(
                "No models available in routing chain. "
                "Add provider API keys and refresh the catalog.",
                code="potato_catalog_empty",
                status=503,
            )

            async def empty() -> AsyncIterator[bytes]:
                yield payload

            return StreamResult(
                status_code=503,
                byte_iter=empty(),
                headers={"content-type": "text/event-stream"},
                key=None,
                model="",
                fallback_index=0,
                decision=decision,
            )

        import asyncio as _aio_s
        import json as _json

        import httpx

        last_status = 503
        last_key = None
        last_model = chain[0]
        last_pid: str | None = None
        saw_ttft_stall = False
        saw_deadline = False
        deadline = self._make_deadline(decision.intent.value)
        chain_budget = max(1, self._max_n_for_intent(decision.intent.value))
        # D1: recovery reserve so last-resort + graceful phases are reachable
        # even when the primary chain consumes its full budget.
        recovery_budget = max(0, min(3, chain_budget - 1))
        max_attempts = chain_budget + recovery_budget
        attempts = 0
        hedges_used = 0
        hedged_consumed: set[str] = set()

        def _error_bytes(
            message: str,
            *,
            code: str,
            status: int = 502,
            retry_after: str | None = None,
        ) -> bytes:
            return frame_sse_error(message, code=code, status=status, retry_after=retry_after)

        for idx, model in enumerate(chain):
            if model.lower() in hedged_consumed:
                continue  # consumed as a parallel-hedge secondary (P2-1)
            remaining = deadline - time.monotonic()
            if remaining < getattr(self.settings, "deadline_guard_seconds", 3.0) and idx > 0:
                last_status, last_model = 504, model
                saw_deadline = True
                logger.warning(
                    "stream request deadline exceeded before model %s (%.1fs left)",
                    model,
                    remaining,
                )
                break
            pid = self._provider_id_for(model)
            try:
                client, upstream_mid = self._client_for(model)
            except Exception as exc:
                # D2: circuit-open skip / unconfigured provider — never a
                # transport failure. The breaker already knows it is open.
                logger.info("stream skip %s: %s; advancing", model, exc)
                if idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    continue
                # Last model's provider unavailable — do NOT fail cold; fall
                # through to the last-resort force-allow + fresh-model retry.
                self._record_outcome(
                    decision,
                    model,
                    None,
                    success=False,
                    status_code=503,
                    intent=decision.intent.value,
                )
                last_status, last_model, last_pid = 503, model, pid
                break
            attempt_body = {**body, "model": upstream_mid}
            # Per-model reasoning_effort normalization (resilience: a reasoning
            # head failing over to a non-reasoning model strips the field).
            from potato.compat import normalize_reasoning_effort as _nre

            attempt_body = _nre(
                attempt_body,
                routed_model=model,
                registry=self.registry,
                default_effort=getattr(self.settings, "default_reasoning_effort", ""),
            )
            t_attempt = time.perf_counter()
            try:
                import asyncio as _aio

                attempt_budget = self._attempt_budget_for(decision.intent.value, remaining)
                status, byte_iter, headers, key = await _aio.wait_for(
                    client.stream(
                        "POST",
                        path,
                        json_body=attempt_body,
                        forward_headers=forward_headers,
                        preferred_key_id=preferred_key_id,
                        max_retries=2,
                        acquire_timeout=min(2.0, attempt_budget),
                    ),
                    timeout=attempt_budget,
                )
            except TimeoutError:
                attempts += 1
                self._circuit_fail(pid, is_transport=False, model_id=model)
                self._record_outcome(
                    decision,
                    model,
                    getattr(key, "key_id", None) if "key" in locals() else None,
                    success=False,
                    status_code=504,
                    intent=decision.intent.value,
                    latency=(time.perf_counter() - t_attempt),
                )
                self._emit_span(
                    self._make_upstream_span(
                        model=model,
                        t0=t_attempt,
                        status=504,
                        success=False,
                        error_message="attempt_deadline_exceeded",
                        span_type="fallback_advance" if idx < len(chain) - 1 else "upstream",
                    )
                )
                if idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(2.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                    logger.info("stream attempt deadline on %s; advancing", model)
                    continue
                last_status, last_model, last_pid = 504, model, pid
                saw_deadline = True
                break
            except (RuntimeError, httpx.HTTPError, OSError) as exc:
                attempts += 1
                # D3: capacity cools the model tier only.
                if isinstance(exc, KeyPoolExhausted):
                    if self.hub is not None:
                        cb = getattr(self.hub, "circuit_breaker", None)
                        if cb is not None:
                            cb.fail(pid or "", is_transport=False, model_id=model)
                else:
                    self._circuit_fail(pid, model_id=model)
                self._record_outcome(
                    decision,
                    model,
                    getattr(key, "key_id", None) if "key" in locals() else None,
                    success=False,
                    status_code=503,
                    intent=decision.intent.value,
                    latency=(time.perf_counter() - t_attempt),
                )
                self._emit_span(
                    self._make_upstream_span(
                        model=model,
                        t0=t_attempt,
                        status=503,
                        success=False,
                        error_message=str(exc),
                        span_type="fallback_advance" if idx < len(chain) - 1 else "upstream",
                    )
                )
                if idx < len(chain) - 1:
                    self.stats.fallback_advances += 1
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(2.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                    logger.info("stream pool/transport on %s: %s; advancing", model, exc)
                    continue
                # Last model transport error — do NOT return cold; fall
                # through to the last-resort force-allow + fresh-model retry.
                last_status, last_model, last_pid = 503, model, pid
                break

            last_status, _last_headers, last_key, last_model, last_pid = (
                status,
                headers,
                key,
                model,
                pid,
            )
            attempts += 1

            if 200 <= status < 300:
                import asyncio

                ct = (headers.get("content-type") or headers.get("Content-Type") or "").lower()
                # Upstream ignored stream:true and returned JSON — convert to SSE (F-18)
                if "application/json" in ct and "text/event-stream" not in ct:
                    raw_parts: list[bytes] = []
                    try:
                        async for chunk in byte_iter:
                            raw_parts.append(chunk)
                            if sum(len(c) for c in raw_parts) > 2_000_000:
                                break
                    except Exception:
                        pass
                    if hasattr(byte_iter, "aclose"):
                        with suppress(Exception):
                            await byte_iter.aclose()
                    raw = b"".join(raw_parts)
                    try:
                        parsed = _json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        parsed = raw.decode("utf-8", errors="replace")
                    sse_payload = json_body_to_sse(parsed, routed_model=model)
                    self._circuit_succeed(pid, model_id=model)
                    self.stats.record(decision.intent.value, model, advanced=idx > 0)

                    async def json_as_sse(p: bytes = sse_payload) -> AsyncIterator[bytes]:
                        yield p

                    return StreamResult(
                        status_code=200,
                        byte_iter=json_as_sse(),
                        headers={**headers, "content-type": "text/event-stream"},
                        key=key,
                        model=model,
                        fallback_index=idx,
                        decision=decision,
                        provider_id=pid,
                    )

                ttft = float(getattr(self.settings, "stream_ttft_timeout_seconds", 12.0) or 12.0)
                # Adaptive TTFT + speculative fail-fast (uniform helper, NMK-405)
                is_last_model = idx == len(chain) - 1
                effective_ttft = self._ttft_budget_for(model, is_last=is_last_model)
                idle = float(getattr(self.settings, "stream_idle_timeout_seconds", 300.0) or 300.0)
                t_stream0 = time.monotonic()
                # P2-1: parallel hedge — race the next (provider-diverse)
                # candidate when the primary is slow. Flag-gated, max 1/request.
                _hedge: dict[str, Any] | None = None
                if (
                    self._hedge_eligible(remaining=remaining, hedges_used=hedges_used)
                    and not is_last_model
                    and idx + 1 < len(chain)
                ):
                    _nxt = chain[idx + 1]
                    _nxt_pid = self._provider_id_for(_nxt)
                    if _nxt_pid != pid and _nxt.lower() not in hedged_consumed:
                        try:
                            _nxt_client, _nxt_mid = self._client_for(_nxt)
                        except Exception:
                            _nxt_client, _nxt_mid = None, ""
                        if _nxt_client is not None:
                            _nxt_body = _nre(
                                {**body, "model": _nxt_mid},
                                routed_model=_nxt,
                                registry=self.registry,
                                default_effort=getattr(
                                    self.settings, "default_reasoning_effort", ""
                                ),
                            )
                            _hedge = {
                                "model": _nxt,
                                "pid": _nxt_pid,
                                "client": _nxt_client,
                                "body": _nxt_body,
                            }
                try:
                    if _hedge is None:
                        first_chunk = await asyncio.wait_for(anext(byte_iter), timeout=effective_ttft)
                    else:
                        hedges_used += 1
                        attempts += 1  # the hedge opens a real upstream stream
                        _h_delay = float(
                            getattr(self.settings, "parallel_hedge_delay_seconds", 2.5) or 2.5
                        )

                        async def _open_secondary(
                            _h=_hedge,
                        ) -> tuple[int, AsyncIterator[bytes], dict[str, str], Any]:
                            _h_budget = max(1.0, min(remaining - _h_delay - 2.0, 45.0))
                            return await asyncio.wait_for(
                                _h["client"].stream(
                                    "POST",
                                    path,
                                    json_body=_h["body"],
                                    forward_headers=forward_headers,
                                    preferred_key_id=preferred_key_id,
                                    max_retries=1,
                                    acquire_timeout=min(2.0, _h_budget),
                                ),
                                timeout=_h_budget,
                            )

                        first_chunk, byte_iter, _winner = await self._hedged_first_chunk(
                            primary_iter=byte_iter,
                            primary_model=model,
                            primary_pid=pid,
                            primary_key=key,
                            secondary_model=_hedge["model"],
                            secondary_pid=_hedge["pid"],
                            open_secondary=_open_secondary,
                            hedge_delay=_h_delay,
                            race_timeout=effective_ttft,
                            decision=decision,
                        )
                        if _winner.get("side") == "secondary":
                            hedged_consumed.add(_hedge["model"].lower())
                            model = _winner["model"]
                            pid = _winner["pid"]
                            key = _winner["key"]
                            headers = _winner["headers"]
                            last_status, last_key, last_model, last_pid = (
                                _winner["status"],
                                key,
                                model,
                                pid,
                            )
                except StopAsyncIteration:
                    # Empty stream body — treat as soft-fail and try next model
                    first_chunk = b""
                    self._circuit_fail(pid, model_id=model)
                    self._emit_span(
                        self._make_upstream_span(
                            model=model,
                            t0=t_attempt,
                            status=502,
                            success=False,
                            error_message="empty_stream",
                            span_type="fallback_advance" if idx < len(chain) - 1 else "upstream",
                        )
                    )
                    if hasattr(byte_iter, "aclose"):
                        with suppress(Exception):
                            await byte_iter.aclose()
                    self.stats.fallback_advances += 1
                    self._record_outcome(
                        decision,
                        model,
                        key.key_id if key else None,
                        success=False,
                        status_code=502,
                        intent=decision.intent.value,
                    )
                    if idx < len(chain) - 1:
                        logger.warning("Empty stream body on %s; falling back", model)
                        continue
                    # Last model: empty stream → terminal error, not 200.
                    # Must continue so we do not fall into the success path.
                    last_status, last_model, last_pid = 502, model, pid
                    logger.warning("Empty stream body on last model %s; returning 502", model)
                    continue
                except TimeoutError:
                    saw_ttft_stall = True
                    last_status = 504
                    logger.warning(
                        "Stream TTFT stalled on %s after %.0fs; falling back",
                        model,
                        ttft,
                    )
                    self._circuit_fail(pid, model_id=model)
                    self._emit_span(
                        self._make_upstream_span(
                            model=model,
                            t0=t_attempt,
                            status=504,
                            success=False,
                            error_message=f"ttft_timeout_{ttft:.0f}s",
                            span_type="fallback_advance",
                            metadata={"ttft_timeout_s": ttft},
                        )
                    )
                    if hasattr(byte_iter, "aclose"):
                        with suppress(Exception):
                            await byte_iter.aclose()
                    self.stats.fallback_advances += 1
                    self._record_outcome(
                        decision,
                        model,
                        key.key_id if key else None,
                        success=False,
                        latency=ttft,
                        status_code=504,
                        intent=decision.intent.value,
                    )
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(2.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                    continue
                except Exception as exc:
                    last_status = 502
                    logger.warning("Stream open failed on %s: %s; falling back", model, exc)
                    self._circuit_fail(pid, model_id=model)
                    self._emit_span(
                        self._make_upstream_span(
                            model=model,
                            t0=t_attempt,
                            status=502,
                            success=False,
                            error_message=str(exc),
                            span_type="fallback_advance",
                        )
                    )
                    if hasattr(byte_iter, "aclose"):
                        with suppress(Exception):
                            await byte_iter.aclose()
                    self.stats.fallback_advances += 1
                    self._record_outcome(
                        decision,
                        model,
                        key.key_id if key else None,
                        success=False,
                        status_code=502,
                        intent=decision.intent.value,
                    )
                    await sleep_backoff(
                        idx,
                        base=self.settings.retry_backoff_base_seconds,
                        cap=min(2.0, self.settings.retry_backoff_cap_seconds),
                        max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                    )
                    continue

                ttft_latency = max(0.01, time.monotonic() - t_stream0)
                self._circuit_succeed(pid, model_id=model)
                self._emit_span(
                    self._make_upstream_span(
                        model=model,
                        t0=t_attempt,
                        status=status,
                        success=True,
                        metadata={"ttft_ms": ttft_latency * 1000, "stream": True},
                    )
                )
                self.stats.record(decision.intent.value, model, advanced=idx > 0)
                # Adaptive: first-token latency feeds speed score immediately
                self._record_outcome(
                    decision,
                    model,
                    key.key_id if key else None,
                    success=True,
                    latency=ttft_latency,
                    status_code=status,
                    intent=decision.intent.value,
                    had_tools=bool(body.get("tools") or body.get("functions")),
                )

                # Bind loop vars so the generator does not close over the last iteration
                bound_model = model
                bound_key_id = key.key_id if key else None
                bound_idle = idle
                bound_t0 = t_stream0
                bound_ttft_ms = ttft_latency * 1000
                usage_bag: dict[str, int] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                }
                # Holder so robust_iter can flip stream_failed after StreamResult exists
                result_holder: dict[str, StreamResult | None] = {"r": None}

                async def robust_iter(
                    first: bytes,
                    rest: AsyncIterator[bytes],
                    *,
                    mid: str = bound_model,
                    kid: str | None = bound_key_id,
                    idle_s: float = bound_idle,
                    t0: float = bound_t0,
                    usage: dict[str, int] = usage_bag,
                ) -> AsyncIterator[bytes]:
                    total_tokens = 0
                    # Bounded queue for backpressure — blocks upstream when client is slow
                    _BP_QUEUE_SIZE = 32
                    _bp_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=_BP_QUEUE_SIZE)
                    _upstream_done = False
                    _upstream_error: Exception | None = None

                    async def _producer() -> None:
                        """Read from upstream and put into bounded queue."""
                        nonlocal _upstream_done, _upstream_error, total_tokens
                        try:
                            async for chunk in rest:
                                is_err = _scan_for_tokens(chunk)
                                if is_err:
                                    break
                                await _bp_queue.put(chunk)
                        except Exception as e:
                            _upstream_error = e
                        finally:
                            _upstream_done = True
                            with suppress(Exception):
                                await _bp_queue.put(None)  # sentinel

                    upstream_error_frame: dict | None = None

                    def _scan_for_tokens(c: bytes) -> bool:
                        nonlocal total_tokens, upstream_error_frame
                        if b'"usage"' in c or b"completion_tokens" in c:
                            import re

                            p = re.search(rb'"prompt_tokens"\s*:\s*(\d+)', c)
                            ct = re.search(rb'"completion_tokens"\s*:\s*(\d+)', c)
                            if p and ct:
                                pt_i, ct_i = int(p.group(1)), int(ct.group(1))
                                total_tokens = pt_i + ct_i
                                usage["prompt_tokens"] = pt_i
                                usage["completion_tokens"] = ct_i
                                self.stats.record_tokens(mid, kid, pt_i, ct_i)
                        # P1-1: reconcile upstream mid-stream error frames
                        # (200 SSE followed by data: {"error": ...}).
                        if b'"error"' in c and upstream_error_frame is None:
                            import json as _scan_json

                            for raw_line in c.split(b"\n"):
                                if not raw_line.startswith(b"data:"):
                                    continue
                                payload = raw_line[5:].strip()
                                if payload in (b"[DONE]", b"") or not payload.startswith(b"{"):
                                    continue
                                try:
                                    obj = _scan_json.loads(payload)
                                except Exception:
                                    continue
                                if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
                                    upstream_error_frame = obj["error"]
                                    return True
                        return upstream_error_frame is not None

                    async def _emit_stream_error(
                        err_msg: str, *, code: str
                    ) -> AsyncIterator[bytes]:
                        finish = {
                            "id": "potato-stream-error",
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "error",
                                }
                            ],
                        }
                        err_evt = openai_error(
                            err_msg,
                            code=code,
                            type_="server_error",
                        )
                        yield (b"data: " + _json.dumps(finish).encode("utf-8") + b"\n\n")
                        yield (b"data: " + _json.dumps(err_evt).encode("utf-8") + b"\n\n")
                        yield b"data: [DONE]\n\n"

                    def _mark_failed() -> None:
                        held = result_holder["r"]
                        if held is not None:
                            held.stream_failed = True
                        self._circuit_fail(pid, model_id=mid)

                    def _mark_failed_model_tier() -> None:
                        held = result_holder["r"]
                        if held is not None:
                            held.stream_failed = True

                    def _cancel_producer() -> None:
                        if not producer_task.done():
                            producer_task.cancel()

                    async def _await_producer() -> None:
                        with suppress(asyncio.CancelledError, Exception):
                            await producer_task

                    # Start producer task
                    producer_task = asyncio.create_task(_producer())
                    try:
                        if first:
                            is_err = _scan_for_tokens(first)
                            if not is_err:
                                yield first
                        while True:
                            try:
                                chunk = await asyncio.wait_for(_bp_queue.get(), timeout=idle_s)
                            except TimeoutError:
                                logger.warning(
                                    "Stream idle timeout on %s after %.0fs — emitting error SSE",
                                    mid,
                                    idle_s,
                                )
                                _cancel_producer()
                                await _await_producer()
                                if hasattr(rest, "aclose"):
                                    with suppress(Exception):
                                        await rest.aclose()
                                elapsed = max(0.01, time.monotonic() - t0)
                                self._record_outcome(
                                    decision,
                                    mid,
                                    kid,
                                    success=False,
                                    latency=elapsed,
                                    tokens=total_tokens or None,
                                    status_code=504,
                                )
                                _mark_failed()
                                async for err_chunk in _emit_stream_error(
                                    f"Stream idle timeout after {idle_s:.0f}s",
                                    code="upstream_stream_idle",
                                ):
                                    yield err_chunk
                                return
                            if chunk is None:  # sentinel from producer
                                break
                            yield chunk
                        # Producer done — check for errors
                        if _upstream_error and not isinstance(_upstream_error, StopAsyncIteration):
                            raise _upstream_error
                        # P1-1: a mid-stream upstream error frame is a model-tier
                        # failure, not a success — even when usage was seen.
                        if upstream_error_frame is not None:
                            elapsed = max(0.01, time.monotonic() - t0)
                            self._record_outcome(
                                decision,
                                mid,
                                kid,
                                success=False,
                                latency=elapsed,
                                tokens=total_tokens or None,
                                status_code=502,
                                intent=decision.intent.value,
                            )
                            if self.hub is not None:
                                cb = getattr(self.hub, "circuit_breaker", None)
                                if cb is not None:
                                    cb.fail(pid or "", is_transport=False, model_id=mid)
                            _mark_failed_model_tier()
                            async for err_chunk in _emit_stream_error(
                                str(upstream_error_frame.get("message") or "Upstream stream error")[:500],
                                code="upstream_stream_error",
                            ):
                                yield err_chunk
                            return
                        # Full stream done — update speed with total time + tokens
                        elapsed = max(0.01, time.monotonic() - t0)
                        if total_tokens > 0:
                            self._record_outcome(
                                decision,
                                mid,
                                kid,
                                success=True,
                                latency=elapsed,
                                tokens=total_tokens,
                                status_code=200,
                            )
                        return
                    except (asyncio.CancelledError, GeneratorExit):
                        _cancel_producer()
                        await _await_producer()
                        if hasattr(rest, "aclose"):
                            with suppress(Exception):
                                await rest.aclose()
                        raise
                    except Exception as e:
                        _cancel_producer()
                        await _await_producer()
                        if hasattr(rest, "aclose"):
                            with suppress(Exception):
                                await rest.aclose()
                        logger.warning(
                            "Stream ended early on %s: %s — closing SSE with error",
                            mid,
                            e,
                        )
                        _mark_failed()
                        # finish_reason=error chunk + error event before [DONE] (F-17)
                        try:
                            async for err_chunk in _emit_stream_error(
                                str(e)[:500],
                                code="upstream_stream_error",
                            ):
                                yield err_chunk
                        except Exception:
                            pass
                        return
                    finally:
                        _cancel_producer()
                        await _await_producer()
                        # D6b: deterministic cleanup — close the upstream
                        # iterator instead of relying on GC finalizers.
                        if hasattr(rest, "aclose"):
                            with suppress(Exception):
                                await rest.aclose()

                stream_result = StreamResult(
                    status_code=status,
                    byte_iter=robust_iter(first_chunk, byte_iter),
                    headers=headers,
                    key=key,
                    model=model,
                    fallback_index=idx,
                    decision=decision,
                    upstream_ttft_ms=bound_ttft_ms,
                    usage=usage_bag,
                    prompt_tokens=usage_bag["prompt_tokens"],
                    completion_tokens=usage_bag["completion_tokens"],
                    cached_tokens=usage_bag["cached_tokens"],
                    provider_id=pid,
                    stream_failed=False,
                )
                result_holder["r"] = stream_result
                return stream_result

            # Failed stream open — advance on same retryable set as JSON path
            # D3: model-tier statuses cool only the model.
            self._record_circuit(pid, model=model, status=status)
            err_raw = b""
            try:
                async for chunk in byte_iter:
                    err_raw += chunk
                    if len(err_raw) > 8192:
                        break
            except Exception:
                pass

            err_body: Any = None
            if err_raw:
                try:
                    err_body = _json.loads(err_raw.decode("utf-8", errors="replace"))
                except Exception:
                    err_body = err_raw.decode("utf-8", errors="replace")
            else:
                ra = headers.get("Retry-After") or headers.get("retry-after")
                err_body = wrap_upstream_error(f"Upstream error HTTP {status}", status=status)
                if ra:
                    err_body = openai_error(
                        f"Upstream error HTTP {status}",
                        code="upstream_error",
                        type_=(
                            "server_error"
                            if status >= 500 or status == 429
                            else "invalid_request_error"
                        ),
                        metadata={"retry_after": ra},
                    )
                err_raw = _json.dumps(err_body).encode("utf-8")

            # Always normalize non-empty upstream error bodies to OpenAI envelope
            if err_body is not None:
                wrapped = wrap_upstream_error(err_body, status=status)
                err_raw = b"data: " + _json.dumps(wrapped).encode("utf-8") + b"\n\ndata: [DONE]\n\n"
            else:
                err_raw = frame_sse_error(
                    f"Upstream error HTTP {status}",
                    code="upstream_error",
                    status=status,
                )

            retryable = status in {401, 403, 404, 405, 408, 429, 500, 502, 503, 504} or (
                status == 400 and _is_retryable_model_error(status, err_body)
            )
            self._record_outcome(
                decision,
                model,
                key.key_id if key else None,
                success=False,
                status_code=status,
                unavailable=status == 404,
                intent=decision.intent.value,
            )
            if retryable:
                if idx < len(chain) - 1:
                    # 504 = gateway timeout — advance immediately, no backoff.
                    # 429 = rate limited — cooldown recorded, advance immediately.
                    if status == 504:
                        pass  # no backoff — advance immediately
                    elif status == 429:
                        pass  # no backoff — advance immediately
                    elif status in {500, 502, 503}:
                        await sleep_backoff(
                            idx,
                            base=self.settings.retry_backoff_base_seconds,
                            cap=min(1.0, self.settings.retry_backoff_cap_seconds),
                            max_delay=max(0.0, (deadline - time.monotonic()) - 2.0),
                        )
                    self.stats.fallback_advances += 1
                    logger.info(
                        "stream model %s failed status=%s; falling back",
                        model,
                        status,
                    )
                    continue
                # Chain exhausted with retryable error: fall through to recovery and graceful fallback!
                self.stats.fallback_advances += 1
                logger.warning(
                    "stream chain exhausted on model %s status=%s; advancing to recovery and graceful fallback",
                    model,
                    status,
                )
                last_status, last_model, last_pid, last_key = status, model, pid, key
                break

            # Non-retryable error (e.g. 400 client syntax error): return immediately
            async def err_bytes(payload: bytes = err_raw) -> AsyncIterator[bytes]:
                yield payload

            self.stats.record(decision.intent.value, model, advanced=idx > 0)
            return StreamResult(
                status_code=status,
                byte_iter=err_bytes(),
                headers={**headers, "content-type": "text/event-stream"},
                key=key,
                model=model,
                fallback_index=idx,
                decision=decision,
                provider_id=pid,
            )

        # ── Last-resort: retry fresh models from the widest pool ──
        # D4: scoped recovery — no global cooldown wipe / force-allow-all.
        # P1-4: recovery runs until 85% of the deadline (see execute_json).
        last_resort_tried: list[str] = []
        phases = self._phase_deadlines(deadline, time.monotonic())
        if time.monotonic() < phases["recovery"] and last_status >= 400 and attempts < max_attempts:
            self.stats.recovery_entered += 1
            tried = {m.lower() for m in chain}
            can_recover_wide = self._is_auto_decision(decision) or getattr(
                self.settings, "allow_graceful_fallback_on_explicit", False
            )
            fresh, _retry_chain = self._recovery_pool(
                decision, had_tools=had_tools, tried=tried,
                allow_any_live=can_recover_wide,
            )
            last_resort_tried = list(fresh)
            self._clear_model_cooldowns(fresh)
            if fresh:
                import asyncio as _aio_s

                logger.warning(
                    "stream last-resort: retrying %s intent-aware models (intent=%s)",
                    len(fresh),
                    decision.intent.value,
                )
                for idx2, model2 in enumerate(fresh):
                    rem2 = deadline - time.monotonic()
                    if rem2 < 3.0:
                        saw_deadline = True
                        break
                    if attempts >= max_attempts:
                        break
                    pid2 = self._provider_id_for(model2)
                    try:
                        client2, upstream_mid2 = self._client_for(model2)
                    except Exception:
                        # D2: skip — never a transport failure.
                        continue
                    attempt_body2 = {**body, "model": upstream_mid2}
                    from potato.compat import normalize_reasoning_effort as _nre2

                    attempt_body2 = _nre2(
                        attempt_body2,
                        routed_model=model2,
                        registry=self.registry,
                        default_effort=getattr(self.settings, "default_reasoning_effort", ""),
                    )
                    if hasattr(self.registry, "ladder"):
                        rec2 = self.registry.ladder.model_recommendations(model2)
                        if rec2:
                            ml2 = rec2.get("max_tokens_limit")
                            if ml2 and not attempt_body2.get("max_tokens"):
                                attempt_body2["max_tokens"] = ml2
                            elif ml2 and attempt_body2.get("max_tokens"):
                                attempt_body2["max_tokens"] = min(
                                    attempt_body2["max_tokens"], ml2
                                )
                    time.perf_counter()
                    try:
                        budget2 = max(1.0, min(rem2, 45.0))
                        s2, byte_iter2, hd2, k2 = await _aio_s.wait_for(
                            client2.stream(
                                "POST",
                                path,
                                json_body=attempt_body2,
                                forward_headers=forward_headers,
                                preferred_key_id=preferred_key_id,
                                max_retries=2,
                                acquire_timeout=min(2.0, budget2),
                            ),
                            timeout=budget2,
                        )
                    except (TimeoutError, RuntimeError, httpx.HTTPError, OSError) as exc2:
                        attempts += 1
                        if isinstance(exc2, KeyPoolExhausted):
                            if self.hub is not None:
                                cb = getattr(self.hub, "circuit_breaker", None)
                                if cb is not None:
                                    cb.fail(pid2 or "", is_transport=False, model_id=model2)
                        else:
                            self._circuit_fail(pid2, model_id=model2)
                        self._record_outcome(
                            decision,
                            model2,
                            None,
                            success=False,
                            status_code=503,
                            intent=decision.intent.value,
                        )
                        continue

                    attempts += 1
                    if 200 <= s2 < 300:
                        ct2 = (hd2.get("content-type") or hd2.get("Content-Type") or "").lower()
                        if "application/json" in ct2 and "text/event-stream" not in ct2:
                            raw_parts: list[bytes] = []
                            try:
                                async for chunk in byte_iter2:
                                    raw_parts.append(chunk)
                                    if sum(len(c) for c in raw_parts) > 2_000_000:
                                        break
                            except Exception:
                                pass
                            if hasattr(byte_iter2, "aclose"):
                                with suppress(Exception):
                                    await byte_iter2.aclose()
                            raw = b"".join(raw_parts)
                            try:
                                parsed = _json.loads(raw.decode("utf-8", errors="replace"))
                            except Exception:
                                parsed = raw.decode("utf-8", errors="replace")
                            sse_payload = json_body_to_sse(parsed, routed_model=model2)
                            self._circuit_succeed(pid2, model_id=model2)
                            self.stats.record(decision.intent.value, model2, advanced=True)

                            async def json_as_sse2(
                                p: bytes = sse_payload,
                            ) -> AsyncIterator[bytes]:
                                yield p

                            return StreamResult(
                                status_code=200,
                                byte_iter=json_as_sse2(),
                                headers={**hd2, "content-type": "text/event-stream"},
                                key=k2,
                                model=model2,
                                fallback_index=len(chain) + idx2,
                                decision=decision,
                                provider_id=pid2,
                            )

                        ttft2 = self._ttft_budget_for(model2, is_last=False)
                        idle2 = float(
                            getattr(self.settings, "stream_idle_timeout_seconds", 300.0) or 300.0
                        )
                        t_stream2 = time.monotonic()
                        try:
                            first_chunk2 = await _aio_s.wait_for(anext(byte_iter2), timeout=ttft2)
                        except (StopAsyncIteration, TimeoutError, Exception):
                            self._circuit_fail(pid2, model_id=model2)
                            if hasattr(byte_iter2, "aclose"):
                                with suppress(Exception):
                                    await byte_iter2.aclose()
                            self._record_outcome(
                                decision,
                                model2,
                                k2.key_id if k2 else None,
                                success=False,
                                status_code=504,
                                intent=decision.intent.value,
                            )
                            continue

                        # Stream opened successfully — return it
                        self._circuit_succeed(pid2, model_id=model2)
                        ttft_lat2 = max(0.01, time.monotonic() - t_stream2)
                        self._record_outcome(
                            decision,
                            model2,
                            k2.key_id if k2 else None,
                            success=True,
                            latency=ttft_lat2,
                            status_code=s2,
                            intent=decision.intent.value,
                            had_tools=had_tools,
                        )
                        self.stats.record(decision.intent.value, model2, advanced=True)
                        bound_model2 = model2
                        bound_key_id2 = k2.key_id if k2 else None
                        bound_idle2 = idle2
                        bound_t02 = t_stream2
                        bound_ttft_ms2 = ttft_lat2 * 1000
                        usage_bag2: dict[str, int] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "cached_tokens": 0,
                        }
                        result_holder2: dict[str, StreamResult | None] = {"r": None}

                        async def robust_iter2(
                            first: bytes,
                            rest: AsyncIterator[bytes],
                            *,
                            mid: str = bound_model2,
                            kid: str | None = bound_key_id2,
                            idle_s: float = bound_idle2,
                            t0: float = bound_t02,
                            usage: dict[str, int] = usage_bag2,
                        ) -> AsyncIterator[bytes]:
                            total_tokens = 0
                            _BP = 32
                            _bpq: _aio_s.Queue[bytes | None] = _aio_s.Queue(maxsize=_BP)
                            _up_done = False
                            _up_err: Exception | None = None

                            async def _prod() -> None:
                                nonlocal _up_done, _up_err, total_tokens
                                try:
                                    async for c in rest:
                                        _scan2(c)
                                        await _bpq.put(c)
                                except Exception as e:
                                    _up_err = e
                                finally:
                                    _up_done = True
                                    with suppress(Exception):
                                        await _bpq.put(None)

                            def _scan2(c: bytes) -> None:
                                nonlocal total_tokens
                                if b'"usage"' in c or b"completion_tokens" in c:
                                    import re

                                    p = re.search(rb'"prompt_tokens"\s*:\s*(\d+)', c)
                                    ct = re.search(rb'"completion_tokens"\s*:\s*(\d+)', c)
                                    if p and ct:
                                        pt_i, ct_i = int(p.group(1)), int(ct.group(1))
                                        total_tokens = pt_i + ct_i
                                        usage["prompt_tokens"] = pt_i
                                        usage["completion_tokens"] = ct_i
                                        self.stats.record_tokens(mid, kid, pt_i, ct_i)

                            prod_task = _aio_s.create_task(_prod())
                            try:
                                if first:
                                    _scan2(first)
                                    yield first
                                while True:
                                    try:
                                        chunk = await _aio_s.wait_for(_bpq.get(), timeout=idle_s)
                                    except TimeoutError:
                                        prod_task.cancel()
                                        with suppress(Exception):
                                            await prod_task
                                        if hasattr(rest, "aclose"):
                                            with suppress(Exception):
                                                await rest.aclose()
                                        elapsed = max(0.01, time.monotonic() - t0)
                                        self._record_outcome(
                                            decision,
                                            mid,
                                            kid,
                                            success=False,
                                            latency=elapsed,
                                            tokens=total_tokens or None,
                                            status_code=504,
                                        )
                                        held = result_holder2["r"]
                                        if held is not None:
                                            held.stream_failed = True
                                        self._circuit_fail(pid2, model_id=mid)
                                        finish = {
                                            "id": "potato-stream-error",
                                            "object": "chat.completion.chunk",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": "error",
                                                }
                                            ],
                                        }
                                        err_evt = openai_error(
                                            f"Stream idle timeout after {idle_s:.0f}s",
                                            code="upstream_stream_idle",
                                            type_="server_error",
                                        )
                                        yield (
                                            b"data: "
                                            + _json.dumps(finish).encode("utf-8")
                                            + b"\n\n"
                                        )
                                        yield (
                                            b"data: "
                                            + _json.dumps(err_evt).encode("utf-8")
                                            + b"\n\n"
                                        )
                                        yield b"data: [DONE]\n\n"
                                        return
                                    if chunk is None:
                                        break
                                    yield chunk
                                if _up_err and not isinstance(_up_err, StopAsyncIteration):
                                    raise _up_err
                                elapsed = max(0.01, time.monotonic() - t0)
                                if total_tokens > 0:
                                    self._record_outcome(
                                        decision,
                                        mid,
                                        kid,
                                        success=True,
                                        latency=elapsed,
                                        tokens=total_tokens,
                                        status_code=200,
                                    )
                                return
                            except (_aio_s.CancelledError, GeneratorExit):
                                prod_task.cancel()
                                with suppress(Exception):
                                    await prod_task
                                if hasattr(rest, "aclose"):
                                    with suppress(Exception):
                                        await rest.aclose()
                                raise
                            except Exception as e:
                                prod_task.cancel()
                                with suppress(Exception):
                                    await prod_task
                                held = result_holder2["r"]
                                if held is not None:
                                    held.stream_failed = True
                                self._circuit_fail(pid2, model_id=mid)
                                try:
                                    finish = {
                                        "id": "potato-stream-error",
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": "error",
                                            }
                                        ],
                                    }
                                    err_evt = openai_error(
                                        str(e)[:500],
                                        code="upstream_stream_error",
                                        type_="server_error",
                                    )
                                    yield (
                                        b"data: " + _json.dumps(finish).encode("utf-8") + b"\n\n"
                                    )
                                    yield (
                                        b"data: " + _json.dumps(err_evt).encode("utf-8") + b"\n\n"
                                    )
                                    yield b"data: [DONE]\n\n"
                                except Exception:
                                    pass
                                return
                            finally:
                                prod_task.cancel()
                                with suppress(Exception):
                                    await prod_task
                                # D6b: deterministic cleanup.
                                if hasattr(rest, "aclose"):
                                    with suppress(Exception):
                                        await rest.aclose()

                        stream_result2 = StreamResult(
                            status_code=s2,
                            byte_iter=robust_iter2(first_chunk2, byte_iter2),
                            headers=hd2,
                            key=k2,
                            model=model2,
                            fallback_index=len(chain) + idx2,
                            decision=decision,
                            upstream_ttft_ms=bound_ttft_ms2,
                            usage=usage_bag2,
                            provider_id=pid2,
                            stream_failed=False,
                        )
                        result_holder2["r"] = stream_result2
                        return stream_result2

                    # Non-2xx stream response
                    # D3: model-tier statuses cool only the model.
                    self._record_circuit(pid2, model=model2, status=s2)
                    err_raw2 = b""
                    try:
                        async for chunk in byte_iter2:
                            err_raw2 += chunk
                            if len(err_raw2) > 8192:
                                break
                    except Exception:
                        pass
                    self._record_outcome(
                        decision,
                        model2,
                        k2.key_id if k2 else None,
                        success=False,
                        status_code=s2,
                        intent=decision.intent.value,
                    )
                    last_status, last_model, last_pid = s2, model2, pid2

        # ── Graceful fallback: try ANY live model from ANY provider ──
        # When the intent-aware chain and fresh retries are exhausted,
        # cast the widest net: every live model whose provider has a
        # runtime. Better to serve with a "wrong-intent" model than 503.
        # Auto decisions only — explicit model requests are never served by
        # a different model — and hard constraints (allowed/free) preserved.
        can_graceful_stream = self._is_auto_decision(decision) or getattr(
            self.settings, "allow_graceful_fallback_on_explicit", False
        )
        if (
            last_status >= 400
            and not saw_deadline
            and can_graceful_stream
        ):
            phases = self._phase_deadlines(deadline, time.monotonic())
            if time.monotonic() < phases["recovery"] and attempts < max_attempts:
                try:
                    any_live = self._any_available_live_models(
                        had_tools=had_tools,
                        intent=decision.intent.value,
                        allowed_models=list(getattr(decision, "allowed_models", None) or []),
                        free_only=(
                            str(getattr(decision, "auto_tier", "") or "").lower() == "free"
                        ),
                    )
                    already = {m.lower() for m in chain} | {m.lower() for m in last_resort_tried}
                    untried_any = [m for m in any_live if m.lower() not in already]
                    untried_any = untried_any[: max(0, max_attempts - attempts)]
                    if untried_any:
                        logger.warning(
                            "stream graceful fallback: trying %s any-provider "
                            "live models (intent=%s, chain exhausted)",
                            len(untried_any),
                            decision.intent.value,
                        )
                        for offset, model_g in enumerate(untried_any):
                            rem_g = deadline - time.monotonic()
                            if rem_g < 3.0:
                                break
                            if attempts >= max_attempts:
                                break
                            pid_g = self._provider_id_for(model_g)
                            try:
                                client_g, upstream_mid_g = self._client_for(model_g)
                            except Exception:
                                # D2: skip — never a transport failure.
                                continue
                            attempt_body_g = {**body, "model": upstream_mid_g}
                            from potato.compat import normalize_reasoning_effort as _nre_g

                            attempt_body_g = _nre_g(
                                attempt_body_g,
                                routed_model=model_g,
                                registry=self.registry,
                                default_effort=getattr(
                                    self.settings, "default_reasoning_effort", ""
                                ),
                            )
                            if hasattr(self.registry, "ladder"):
                                rec_g = self.registry.ladder.model_recommendations(model_g)
                                if rec_g:
                                    ml_g = rec_g.get("max_tokens_limit")
                                    if ml_g and not attempt_body_g.get("max_tokens"):
                                        attempt_body_g["max_tokens"] = ml_g
                                    elif ml_g and attempt_body_g.get("max_tokens"):
                                        attempt_body_g["max_tokens"] = min(
                                            attempt_body_g["max_tokens"], ml_g
                                        )
                            try:
                                budget_g = max(1.0, min(rem_g, 45.0))
                                s_g, byte_iter_g, hd_g, k_g = await _aio_s.wait_for(
                                    client_g.stream(
                                        "POST",
                                        path,
                                        json_body=attempt_body_g,
                                        forward_headers=forward_headers,
                                        preferred_key_id=preferred_key_id,
                                        max_retries=2,
                                        acquire_timeout=min(2.0, budget_g),
                                    ),
                                    timeout=budget_g,
                                )
                            except (TimeoutError, RuntimeError, httpx.HTTPError, OSError) as exc_g:
                                attempts += 1
                                if isinstance(exc_g, KeyPoolExhausted):
                                    if self.hub is not None:
                                        cb = getattr(self.hub, "circuit_breaker", None)
                                        if cb is not None:
                                            cb.fail(pid_g or "", is_transport=False, model_id=model_g)
                                else:
                                    self._circuit_fail(pid_g, model_id=model_g)
                                self._record_outcome(
                                    decision,
                                    model_g,
                                    None,
                                    success=False,
                                    status_code=503,
                                    intent=decision.intent.value,
                                )
                                continue

                            attempts += 1
                            if 200 <= s_g < 300:
                                ct_g = (hd_g.get("content-type") or hd_g.get("Content-Type") or "").lower()
                                if "application/json" in ct_g and "text/event-stream" not in ct_g:
                                    raw_parts_g: list[bytes] = []
                                    try:
                                        async for chunk in byte_iter_g:
                                            raw_parts_g.append(chunk)
                                            if sum(len(c) for c in raw_parts_g) > 2_000_000:
                                                break
                                    except Exception:
                                        pass
                                    if hasattr(byte_iter_g, "aclose"):
                                        with suppress(Exception):
                                            await byte_iter_g.aclose()
                                    raw_g = b"".join(raw_parts_g)
                                    try:
                                        parsed_g = _json.loads(raw_g.decode("utf-8", errors="replace"))
                                    except Exception:
                                        parsed_g = raw_g.decode("utf-8", errors="replace")
                                    sse_payload_g = json_body_to_sse(parsed_g, routed_model=model_g)
                                    self._circuit_succeed(pid_g, model_id=model_g)
                                    self.stats.record(decision.intent.value, model_g, advanced=True)

                                    async def json_as_sse_g(p: bytes = sse_payload_g) -> AsyncIterator[bytes]:
                                        yield p

                                    return StreamResult(
                                        status_code=200,
                                        byte_iter=json_as_sse_g(),
                                        headers={**hd_g, "content-type": "text/event-stream"},
                                        key=k_g,
                                        model=model_g,
                                        fallback_index=len(chain) + offset,
                                        decision=decision,
                                        provider_id=pid_g,
                                    )

                                ttft_g = self._ttft_budget_for(model_g, is_last=False)
                                t_stream_g = time.monotonic()
                                try:
                                    first_chunk_g = await _aio_s.wait_for(anext(byte_iter_g), timeout=ttft_g)
                                except (StopAsyncIteration, TimeoutError, Exception):
                                    self._circuit_fail(pid_g, model_id=model_g)
                                    if hasattr(byte_iter_g, "aclose"):
                                        with suppress(Exception):
                                            await byte_iter_g.aclose()
                                    self._record_outcome(
                                        decision,
                                        model_g,
                                        k_g.key_id if k_g else None,
                                        success=False,
                                        status_code=504,
                                        intent=decision.intent.value,
                                    )
                                    continue

                                self._circuit_succeed(pid_g, model_id=model_g)
                                ttft_lat_g = max(0.01, time.monotonic() - t_stream_g)
                                self._record_outcome(
                                    decision,
                                    model_g,
                                    k_g.key_id if k_g else None,
                                    success=True,
                                    latency=ttft_lat_g,
                                    status_code=s_g,
                                    intent=decision.intent.value,
                                    had_tools=had_tools,
                                )
                                self.stats.record(decision.intent.value, model_g, advanced=True)
                                bound_model_g = model_g
                                bound_key_id_g = k_g.key_id if k_g else None
                                bound_idle_g = float(
                                    getattr(self.settings, "stream_idle_timeout_seconds", 300.0) or 300.0
                                )
                                bound_t0_g = t_stream_g
                                bound_ttft_ms_g = ttft_lat_g * 1000
                                usage_bag_g: dict[str, int] = {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "cached_tokens": 0,
                                }
                                result_holder_g: dict[str, StreamResult | None] = {"r": None}

                                async def robust_iter_g(
                                    first: bytes,
                                    rest: AsyncIterator[bytes],
                                    *,
                                    mid: str = bound_model_g,
                                    kid: str | None = bound_key_id_g,
                                    idle_s: float = bound_idle_g,
                                    t0: float = bound_t0_g,
                                    usage: dict[str, int] = usage_bag_g,
                                ) -> AsyncIterator[bytes]:
                                    total_tokens = 0
                                    _BP_G = 32
                                    _bpq_g: _aio_s.Queue[bytes | None] = _aio_s.Queue(maxsize=_BP_G)
                                    _up_done_g = False
                                    _up_err_g: Exception | None = None

                                    async def _prod_g() -> None:
                                        nonlocal _up_done_g, _up_err_g, total_tokens
                                        try:
                                            async for c in rest:
                                                _scan_g(c)
                                                await _bpq_g.put(c)
                                        except Exception as e:
                                            _up_err_g = e
                                        finally:
                                            _up_done_g = True
                                            with suppress(Exception):
                                                await _bpq_g.put(None)

                                    def _scan_g(c: bytes) -> None:
                                        nonlocal total_tokens
                                        if b'"usage"' in c or b"completion_tokens" in c:
                                            import re
                                            p = re.search(rb'"prompt_tokens"\s*:\s*(\d+)', c)
                                            ct = re.search(rb'"completion_tokens"\s*:\s*(\d+)', c)
                                            if p and ct:
                                                pt_i, ct_i = int(p.group(1)), int(ct.group(1))
                                                total_tokens = pt_i + ct_i
                                                usage["prompt_tokens"] = pt_i
                                                usage["completion_tokens"] = ct_i
                                                self.stats.record_tokens(mid, kid, pt_i, ct_i)

                                    prod_task_g = _aio_s.create_task(_prod_g())
                                    try:
                                        if first:
                                            _scan_g(first)
                                            yield first
                                        while True:
                                            try:
                                                chunk = await _aio_s.wait_for(_bpq_g.get(), timeout=idle_s)
                                            except TimeoutError:
                                                prod_task_g.cancel()
                                                with suppress(Exception):
                                                    await prod_task_g
                                                if hasattr(rest, "aclose"):
                                                    with suppress(Exception):
                                                        await rest.aclose()
                                                elapsed = max(0.01, time.monotonic() - t0)
                                                self._record_outcome(
                                                    decision, mid, kid, success=False,
                                                    latency=elapsed, tokens=total_tokens or None,
                                                    status_code=504,
                                                )
                                                held = result_holder_g["r"]
                                                if held is not None:
                                                    held.stream_failed = True
                                                self._circuit_fail(pid_g, model_id=mid)
                                                finish = {"id": "potato-stream-error", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
                                                err_evt = openai_error(f"Stream idle timeout after {idle_s:.0f}s", code="upstream_stream_idle", type_="server_error")
                                                yield b"data: " + _json.dumps(finish).encode("utf-8") + b"\n\n"
                                                yield b"data: " + _json.dumps(err_evt).encode("utf-8") + b"\n\n"
                                                yield b"data: [DONE]\n\n"
                                                return
                                            if chunk is None:
                                                break
                                            yield chunk
                                        if _up_err_g and not isinstance(_up_err_g, StopAsyncIteration):
                                            raise _up_err_g
                                        elapsed = max(0.01, time.monotonic() - t0)
                                        if total_tokens > 0:
                                            self._record_outcome(
                                                decision, mid, kid, success=True,
                                                latency=elapsed, tokens=total_tokens,
                                                status_code=200,
                                            )
                                        return
                                    except (_aio_s.CancelledError, GeneratorExit):
                                        prod_task_g.cancel()
                                        with suppress(Exception):
                                            await prod_task_g
                                        if hasattr(rest, "aclose"):
                                            with suppress(Exception):
                                                await rest.aclose()
                                        raise
                                    except Exception as e:
                                        prod_task_g.cancel()
                                        with suppress(Exception):
                                            await prod_task_g
                                        held = result_holder_g["r"]
                                        if held is not None:
                                            held.stream_failed = True
                                        self._circuit_fail(pid_g, model_id=mid)
                                        try:
                                            finish = {"id": "potato-stream-error", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
                                            err_evt = openai_error(str(e)[:500], code="upstream_stream_error", type_="server_error")
                                            yield b"data: " + _json.dumps(finish).encode("utf-8") + b"\n\n"
                                            yield b"data: " + _json.dumps(err_evt).encode("utf-8") + b"\n\n"
                                            yield b"data: [DONE]\n\n"
                                        except Exception:
                                            pass
                                        return
                                    finally:
                                        prod_task_g.cancel()
                                        with suppress(Exception):
                                            await prod_task_g
                                        # D6b: deterministic cleanup.
                                        if hasattr(rest, "aclose"):
                                            with suppress(Exception):
                                                await rest.aclose()

                                stream_result_g = StreamResult(
                                    status_code=s_g,
                                    byte_iter=robust_iter_g(first_chunk_g, byte_iter_g),
                                    headers=hd_g,
                                    key=k_g,
                                    model=model_g,
                                    fallback_index=len(chain) + offset,
                                    decision=decision,
                                    upstream_ttft_ms=bound_ttft_ms_g,
                                    usage=usage_bag_g,
                                    provider_id=pid_g,
                                    stream_failed=False,
                                )
                                result_holder_g["r"] = stream_result_g
                                return stream_result_g

                            # Non-2xx — record and try next
                            # D3: model-tier statuses cool only the model.
                            self._record_circuit(pid_g, model=model_g, status=s_g)
                            try:
                                async for _chunk in byte_iter_g:
                                    pass
                            except Exception:
                                pass
                            self._record_outcome(
                                decision, model_g, k_g.key_id if k_g else None,
                                success=False, status_code=s_g,
                                intent=decision.intent.value,
                            )
                            last_status, last_model, last_pid = s_g, model_g, pid_g
                except Exception:
                    logger.exception("stream graceful fallback failed")

        # No stream successfully relayed — never return 2xx with empty body (F-05)
        if saw_deadline:
            terminal_status = 504
            code = "request_deadline_exceeded"
            msg = "Request deadline exceeded before a stream could be opened."
        elif saw_ttft_stall or last_status < 400:
            terminal_status = 504
            code = "upstream_timeout"
            msg = "All models timed out waiting for the first stream token."
        else:
            terminal_status = last_status if last_status >= 400 else 504
            code = "all_providers_failed"
            msg = "All models in routing chain failed to open a stream after recovery."
        if terminal_status < 400:
            terminal_status = 504
        # D5d: terminal SSE errors carry a retry hint for 503s.
        retry_after = "3" if terminal_status == 503 else None
        payload = _error_bytes(msg, code=code, status=terminal_status, retry_after=retry_after)
        self.stats.chain_exhausted += 1

        async def empty_fail(p: bytes = payload) -> AsyncIterator[bytes]:
            yield p

        headers_out: dict[str, str] = {"content-type": "text/event-stream"}
        if retry_after:
            headers_out["Retry-After"] = retry_after
        return StreamResult(
            status_code=terminal_status,
            byte_iter=empty_fail(),
            headers=headers_out,
            key=last_key,
            model=last_model,
            fallback_index=max(0, len(chain) - 1),
            decision=decision,
            provider_id=last_pid,
        )
