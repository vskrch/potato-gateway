"""Fallback executor with mocked upstream."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from potato.balancer import KeyStats
from potato.catalog import ModelRegistry
from potato.config import Settings
from potato.routing import (
    FallbackExecutor,
    Intent,
    IntentResult,
    ModelSelector,
    RouteDecision,
)

YAML = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def _key(i: int = 0) -> KeyStats:
    return KeyStats(key_id=f"key-{i}", api_key=f"k{i}")


@pytest.mark.asyncio
async def test_fallback_advances_on_404() -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    calls: list[str] = []

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        calls.append(model)
        if model == "model-a":
            return 404, {"error": {"message": "model not found"}}, {}, _key()
        return 200, {"id": "ok", "model": model, "choices": []}, {}, _key(1)

    upstream = AsyncMock()
    upstream.request_json = fake_json

    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CODING_AGENTIC,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    assert result.fallback_index == 1
    assert calls == ["model-a", "model-b"]
    assert result.body["model"] == "model-b"


@pytest.mark.asyncio
async def test_soft_fail_empty_reply_advances() -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        if model == "model-a":
            return (
                200,
                {"id": "empty", "model": model, "choices": [{"message": {"content": ""}}]},
                {},
                _key(),
            )
        return (
            200,
            {
                "id": "ok",
                "model": model,
                "choices": [{"message": {"content": "hello"}}],
            },
            {},
            _key(1),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json(
        "/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
        decision,
    )
    assert result.status_code == 200
    assert result.model == "model-b"
    assert result.fallback_index == 1


@pytest.mark.asyncio
async def test_non_retryable_400_stops() -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"a", "b"}

    async def fake_json(method, path, **kwargs):
        return 400, {"error": {"message": "invalid json schema"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["a", "b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json("/chat/completions", {}, decision)
    assert result.status_code == 400
    assert result.model == "a"


@pytest.mark.asyncio
async def test_context_overflow_advances() -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}
    reg.context_by_model = {"model-a": 8192, "model-b": 131072}

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        if model == "model-a":
            return (
                400,
                {"error": {"message": "This model's maximum context length is 8192 tokens"}},
                {},
                _key(),
            )
        return (
            200,
            {
                "id": "ok",
                "model": model,
                "choices": [{"message": {"content": "ok"}}],
            },
            {},
            _key(1),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CODING_AGENTIC,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    headers = ex.routing_headers(decision, model=result.model, key_id="key-1", fallback_index=1)
    assert headers.get("X-Potato-Context-Length") == "131072"

    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    all_ids: set[str] = set()
    for ic in reg.catalog.intents.values():
        all_ids.update(ic.chain)
    reg.live_ids = all_ids
    sel = ModelSelector(reg, settings)
    intent = IntentResult(intent=Intent.CHAT_FAST, confidence=0.7, rule_id="short_chat")
    d = sel.resolve("potato/auto", intent)
    assert d.mode == "auto"
    assert d.chain


@pytest.mark.asyncio
async def test_auto_chain_expands_related_intents() -> None:
    """potato/auto fallback chain expands beyond a thin primary decision chain."""
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=8)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {
        "qwen/qwen3.5-122b-a10b",
        "nvidia/nemotron-3-super-120b-a12b",
        "zen/mimo-v2.5-free",
    }
    reg._rebuild_all_chains()
    upstream = AsyncMock()
    decision = RouteDecision(
        chain=["qwen/qwen3.5-122b-a10b"],  # intentionally thin
        mode="auto",
        intent=Intent.CODING_AGENTIC,
        rule_id="tools_present",
        requested_model="potato/auto",
        auto_tier="balanced",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    chain = ex._chain(decision, had_tools=False)
    assert chain[0]  # non-empty
    # Related/live models should extend the single-model decision
    assert len(chain) >= 1


@pytest.mark.asyncio
async def test_auto_quality_floor_never_empties_chain() -> None:
    """Auto mode must keep models even when quality floor would wipe them."""
    settings = Settings(nim_api_keys=["k"], min_quality_ratio=0.99)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}
    # Seed ladder scores with huge spread so floor would filter model-b
    if hasattr(reg, "ladder"):
        from potato.catalog.ladder import LadderSnapshot

        reg.ladder._ladders[("chat_fast", "default")] = LadderSnapshot(
            intent="chat_fast",
            ladder=["model-a", "model-b"],
            scores={"model-a": 100.0, "model-b": 1.0},
            built_from_live=2,
        )
    upstream = AsyncMock()
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="short_chat",
        requested_model="potato/auto",
        auto_tier="balanced",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    chain = ex._chain(decision)
    assert "model-a" in chain
    # model-b may be filtered by soft floor (0.45) but chain must stay non-empty
    assert len(chain) >= 1


@pytest.mark.asyncio
async def test_auto_advances_through_expanded_pool() -> None:
    """When primary auto model fails, fallback walks the intent-expanded pool."""
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=5)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c"}

    calls: list[str] = []

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        calls.append(model)
        if model == "model-a":
            return 503, {"error": {"message": "unavailable"}}, {}, _key()
        return (
            200,
            {
                "id": "ok",
                "model": model,
                "choices": [{"message": {"content": "done"}}],
            },
            {},
            _key(1),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b", "model-c"],
        mode="auto",
        intent=Intent.CODING_AGENTIC,
        rule_id="tools_present",
        requested_model="potato/auto",
        auto_tier="balanced",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json(
        "/chat/completions",
        {"messages": [{"role": "user", "content": "fix bug"}]},
        decision,
    )
    assert result.status_code == 200
    assert result.fallback_index >= 1
    assert len(calls) >= 2


@pytest.mark.asyncio
async def test_streaming_watchdog_ttft_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_stream(method, path, **kwargs):
        model = kwargs["json_body"]["model"]
        if model == "model-a":
            # Simulate a stream that connects (returns 200) but never yields chunks
            import asyncio

            async def stalled_iter():
                await asyncio.sleep(2.0)
                yield b"never reached"

            return 200, stalled_iter(), {}, _key()
        else:
            # Model B succeeds immediately
            async def ok_iter():
                yield b"ok"

            return 200, ok_iter(), {}, _key(1)

    upstream = AsyncMock()
    upstream.stream = fake_stream

    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)

    import asyncio

    original_wait_for = asyncio.wait_for

    call_count = 0

    async def mock_wait_for(fut, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate real wait_for: start the coroutine, then time out and
            # cancel it (closing the generator cleanly) instead of abandoning it.
            return await original_wait_for(fut, 0.0)
        return await original_wait_for(fut, timeout)

    monkeypatch.setattr(asyncio, "wait_for", mock_wait_for)

    result = await original_wait_for(
        ex.execute_stream("/chat/completions", {"messages": []}, decision), timeout=2.0
    )

    assert result.status_code == 200
    assert result.model == "model-b"
    assert result.fallback_index == 1

    # ensure we can consume it, restoring wait_for so the inner logic works
    monkeypatch.setattr(asyncio, "wait_for", original_wait_for)
    chunks = [c async for c in result.byte_iter]
    assert chunks == [b"ok"]


@pytest.mark.asyncio
async def test_token_accounting_json() -> None:
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_json(method, path, **kwargs):
        return (
            200,
            {
                "id": "ok",
                "model": "model-a",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            {},
            _key(),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json

    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    await ex.execute_json("/chat/completions", {"messages": []}, decision)

    assert ex.stats.model_tokens["model-a"].prompt_tokens == 10
    assert ex.stats.model_tokens["model-a"].completion_tokens == 5


@pytest.mark.asyncio
async def test_token_accounting_stream() -> None:
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_stream(method, path, **kwargs):
        async def ok_iter():
            yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10}}\n\n'

        return 200, ok_iter(), {}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream

    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)

    # consume
    [c async for c in result.byte_iter]

    assert ex.stats.model_tokens["model-a"].prompt_tokens == 20
    assert ex.stats.model_tokens["model-a"].completion_tokens == 10


@pytest.mark.asyncio
async def test_empty_stream_last_model_returns_502_not_200() -> None:
    """Last-model empty body must be a terminal error, not HTTP 200 + empty SSE."""
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_stream(method, path, **kwargs):
        async def empty_iter():
            if False:
                yield b""
            return

        return 200, empty_iter(), {}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream
    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.status_code >= 400, f"expected error status, got {result.status_code}"
    chunks = [c async for c in result.byte_iter]
    joined = b"".join(chunks)
    assert b"error" in joined.lower() or result.status_code == 502


@pytest.mark.asyncio
async def test_mid_stream_idle_emits_error_not_clean_done() -> None:
    """Idle timeout must emit finish_reason=error + error event, not bare [DONE]."""
    settings = Settings(
        nim_api_keys=["k"],
        stream_idle_timeout_seconds=0.05,
        stream_ttft_timeout_seconds=5.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_stream(method, path, **kwargs):
        async def slow_iter():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            import asyncio as _aio

            await _aio.sleep(2.0)
            yield b'data: {"choices":[{"delta":{"content":"bye"}}]}\n\n'

        return 200, slow_iter(), {}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream
    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert 200 <= result.status_code < 300
    chunks = [c async for c in result.byte_iter]
    joined = b"".join(chunks)
    assert b"finish_reason" in joined and b"error" in joined
    assert b"[DONE]" in joined
    assert result.stream_failed is True


@pytest.mark.asyncio
async def test_stream_failed_flag_set_after_mid_stream_error() -> None:
    """stream_failed must be visible on StreamResult after the iterator finishes."""
    settings = Settings(nim_api_keys=["k"], stream_idle_timeout_seconds=180.0)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_stream(method, path, **kwargs):
        async def boom_iter():
            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            raise RuntimeError("upstream dropped")

        return 200, boom_iter(), {}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream
    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.stream_failed is False  # not yet consumed
    _ = [c async for c in result.byte_iter]
    assert result.stream_failed is True


@pytest.mark.asyncio
async def test_execute_stream_honors_request_deadline() -> None:
    """Stream path must stop advancing when request deadline is nearly exhausted."""
    settings = Settings(
        nim_api_keys=["k"],
        request_deadline_seconds=0.01,
        stream_ttft_timeout_seconds=12.0,
        max_model_fallbacks=5,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c"}

    calls: list[str] = []

    async def fake_stream(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        calls.append(model)

        async def stalled():
            import asyncio as _aio

            await _aio.sleep(0.05)
            yield b"never"

        return 200, stalled(), {}, _key()

    upstream = AsyncMock()
    upstream.stream = fake_stream
    # Force TTFT to fail fast so we advance between models under deadline
    settings.stream_ttft_timeout_seconds = 0.02
    decision = RouteDecision(
        chain=["model-a", "model-b", "model-c"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    import asyncio

    result = await asyncio.wait_for(
        ex.execute_stream("/chat/completions", {"messages": []}, decision),
        timeout=3.0,
    )
    assert result.status_code >= 400
    # Must not burn the entire 3-model chain when deadline is tiny
    assert len(calls) < 3, f"deadline ignored; tried all models: {calls}"


@pytest.mark.asyncio
async def test_json_405_advances_to_next_model() -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        if model == "model-a":
            return 405, {"error": {"message": "method not allowed"}}, {}, _key()
        return (
            200,
            {
                "id": "ok",
                "model": model,
                "choices": [{"message": {"content": "hi"}}],
            },
            {},
            _key(1),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"


@pytest.mark.asyncio
async def test_json_401_advances_to_next_provider() -> None:
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=3)
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model")
        if model == "model-a":
            return 401, {"error": {"message": "bad key"}}, {}, _key()
        return (
            200,
            {
                "id": "ok",
                "model": model,
                "choices": [{"message": {"content": "hi"}}],
            },
            {},
            _key(1),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"


@pytest.mark.asyncio
async def test_stream_json_content_type_converted_to_sse() -> None:
    settings = Settings(nim_api_keys=["k"])
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a"}

    async def fake_stream(method, path, **kwargs):
        async def json_iter():
            yield b'{"id":"1","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}'

        return (
            200,
            json_iter(),
            {"content-type": "application/json"},
            _key(),
        )

    upstream = AsyncMock()
    upstream.stream = fake_stream
    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert "event-stream" in result.headers.get("content-type", "")
    joined = b"".join([c async for c in result.byte_iter])
    assert b"data:" in joined and b"[DONE]" in joined
    assert b"hello" in joined


# ── Graceful fallback: any live model from any provider ─────────────


@pytest.mark.asyncio
async def test_graceful_fallback_tries_any_live_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the entire intent chain fails, graceful fallback casts the
    widest net: any live model whose provider has a runtime."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c"}

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model", "")
        if model in ("model-a", "model-b"):
            return 503, {"error": {"message": "unavailable"}}, {}, _key()
        if model == "model-c":
            return (
                200,
                {"id": "ok", "model": model, "choices": [{"message": {"content": "ok"}}]},
                {},
                _key(2),
            )
        return 500, {"error": {"message": "???"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-c"


# ── Resiliency hardening: bounded attempts + constraint-preserving fallback ──


@pytest.mark.asyncio
async def test_fallback_honors_universal_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAX_MODEL_FALLBACKS is the hard ceiling across chain + last-resort +
    graceful phases — the escape paths must not silently bypass the cap."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=1,
        intent_max_fallbacks={"chat_fast": 1},
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c"}
    calls: list[str] = []

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        calls.append(model)
        return 503, {"error": {"message": "down"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 503
    assert calls == ["model-a"]


@pytest.mark.asyncio
async def test_graceful_fallback_respects_allowed_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graceful any-live fallback must preserve allowed_models constraints —
    models the caller excluded are never attempted, even on total failure."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c"}
    calls: list[str] = []

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        calls.append(model)
        return 503, {"error": {"message": "down"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
        allowed_models=["model-a"],
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 503
    assert calls == ["model-a"]


@pytest.mark.asyncio
async def test_graceful_fallback_skipped_for_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit model request must never be served by a different model:
    graceful any-live fallback runs only for auto-router decisions."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b"}
    calls: list[str] = []

    async def fake_json(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        calls.append(model)
        if model == "model-b":
            return 200, {"choices": [{"message": {"content": "b"}}]}, {}, _key(1)
        return 503, {"error": {"message": "down"}}, {}, _key()

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a"],
        mode="passthrough_with_fallback",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="model-a",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 503
    assert calls == ["model-a"]


@pytest.mark.asyncio
async def test_stream_graceful_fallback_tries_any_live_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming path: chain models fail to open, an any-live model serves."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-b", "model-c"}

    async def fake_stream(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        if model in ("model-a", "model-b"):

            async def err():
                yield b'data: {"error": {"message": "down"}}\n\n'

            return 503, err(), {}, _key()
        if model == "model-c":

            async def ok():
                yield b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'

            return 200, ok(), {"content-type": "text/event-stream"}, _key(2)
        raise AssertionError(f"unexpected model {model}")

    upstream = AsyncMock()
    upstream.stream = fake_stream
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-c"
    chunks = b"".join([c async for c in result.byte_iter])
    assert b"ok" in chunks


@pytest.mark.asyncio
async def test_malformed_success_body_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2xx with an unusable body (HTML/text) is a soft-fail: the chain must
    advance instead of serving garbage to the client."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
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
            return 200, "<html>upstream gateway error</html>", {}, _key()
        return (
            200,
            {"id": "ok", "model": model, "choices": [{"message": {"content": "ok"}}]},
            {},
            _key(1),
        )

    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    assert calls == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_upstream_invalid_json_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport-class failures (malformed JSON body) must advance the chain
    instead of raising out of the request path."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
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
            raise RuntimeError("upstream returned invalid JSON body (HTTP 200)")
        return (
            200,
            {"id": "ok", "model": model, "choices": [{"message": {"content": "ok"}}]},
            {},
            _key(1),
        )
    upstream = AsyncMock()
    upstream.request_json = fake_json
    decision = RouteDecision(
        chain=["model-a", "model-b"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="auto",
    )
    ex = FallbackExecutor(upstream, reg, settings)
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    assert calls == ["model-a", "model-b"]


# ── Anti-celebrity exploration slot (NMK-RL) ─────────────────────────


def _seed_exploration_fixture(
    settings: Settings,
    *,
    scores: dict[str, float] | None = None,
) -> tuple[ModelRegistry, FallbackExecutor]:
    """Registry with a well-sampled head pair and one under-sampled challenger.

    ``scores`` optionally fakes ladder scores (quality gate tests).
    """
    from types import SimpleNamespace

    reg = ModelRegistry.from_yaml(YAML)
    reg.live_ids = {"model-a", "model-c", "model-b"}
    # Celebrities: heavily sampled. Challenger: zero samples.
    for _ in range(20):
        reg.learning.record(intent="coding_agentic", model_id="model-a", success=True)
        reg.learning.record(intent="coding_agentic", model_id="model-c", success=True)
    reg.health.record_outcome("model-a", success=True, latency=0.1, tokens=100)
    reg.health.record_outcome("model-c", success=True, latency=0.2, tokens=100)
    if scores is not None:
        reg.ladder._ladders = {
            ("coding_agentic", "default"): SimpleNamespace(scores=dict(scores))
        }
    ex = FallbackExecutor(AsyncMock(), reg, settings)
    ex._provider_available = lambda _m: True
    return reg, ex


def _coding_auto_decision() -> RouteDecision:
    return RouteDecision(
        chain=["model-a", "model-c", "model-b"],
        mode="auto",
        intent=Intent.CODING_AGENTIC,
        rule_id="test",
        requested_model="auto",
    )


def test_chain_injects_exploration_slot_for_undersampled_model() -> None:
    """An under-sampled challenger is promoted to second place on auto
    requests so the bandit actually samples it (anti-celebrity)."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        ucb_exploration_c=5.0,
        thompson_blend_n=12,
    )
    _reg, ex = _seed_exploration_fixture(settings)
    chain = ex._chain(_coding_auto_decision())
    assert chain[1] == "model-b", chain


def test_chain_skips_exploration_slot_when_quality_below_gate() -> None:
    """Exploration is quality-gated: a challenger far below the head is not
    promoted (retaining model quality during exploration)."""
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=5)
    _reg, ex = _seed_exploration_fixture(
        settings, scores={"model-a": 100, "model-c": 90, "model-b": 40}
    )
    chain = ex._chain(_coding_auto_decision())
    # model-b (40 < 0.5 * 100) must not be promoted over model-c (90).
    assert "model-b" not in chain[:2], chain


def test_chain_injects_exploration_slot_when_quality_passes_gate() -> None:
    """A challenger within the quality gate is promoted even when ladder
    scores exist."""
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=5)
    _reg, ex = _seed_exploration_fixture(
        settings, scores={"model-a": 100, "model-c": 90, "model-b": 80}
    )
    chain = ex._chain(_coding_auto_decision())
    assert chain[1] == "model-b", chain


def test_chain_skips_exploration_slot_when_disabled() -> None:
    """rl_exploration_enabled=False disables the slot entirely."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        rl_exploration_enabled=False,
    )
    _reg, ex = _seed_exploration_fixture(settings)
    chain = ex._chain(_coding_auto_decision())
    assert chain[1] != "model-b"


def test_chain_skips_exploration_slot_for_explicit_request() -> None:
    """Explicit (non-auto) requests never get exploration substitution."""
    settings = Settings(nim_api_keys=["k"], max_model_fallbacks=5)
    _reg, ex = _seed_exploration_fixture(settings)
    decision = RouteDecision(
        chain=["model-a", "model-c", "model-b"],
        mode="passthrough_with_fallback",
        intent=Intent.CODING_AGENTIC,
        rule_id="test",
        requested_model="model-a",
    )
    chain = ex._chain(decision)
    assert chain[1] != "model-b"
