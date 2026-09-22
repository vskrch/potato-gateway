"""Regression tests for high-availability routing and resilience fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
import json
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
from potato.safety.circuit_breaker import ProviderCircuitBreaker, BreakerState

YAML = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def _key(i: int = 0) -> KeyStats:
    return KeyStats(key_id=f"key-{i}", api_key=f"k{i}")


@pytest.mark.asyncio
async def test_stream_fallback_exhaustion_advances_to_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """When all primary models fail in a stream, it must advance to recovery / fresh models instead of returning 503 immediately."""
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

    async def fake_stream(method, path, **kwargs):
        model = (kwargs.get("json_body") or {}).get("model", "")
        calls.append(model)
        if model == "model-b":
            async def ok_iter():
                payload = json.dumps({"choices": [{"delta": {"content": "hi"}}]})
                yield f"data: {payload}\n\n".encode("utf-8")
            return 200, ok_iter(), {"content-type": "text/event-stream"}, _key(1)
        async def err_iter():
            yield json.dumps({"error": {"message": "model down"}}).encode("utf-8")
        return 503, err_iter(), {"content-type": "application/json"}, _key(0)

    mock_client = AsyncMock()
    mock_client.stream = fake_stream

    decision = RouteDecision(
        chain=["model-a"],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="potato/auto",
    )
    ex = FallbackExecutor(mock_client, reg, settings)
    monkeypatch.setattr(ex, "_client_for", lambda m: (mock_client, m))
    monkeypatch.setattr(ex, "_provider_available", lambda _m: True)
    monkeypatch.setattr(ex, "_chain", lambda d, had_tools=False: ["model-a", "model-b"])

    result = await ex.execute_stream("/chat/completions", {"messages": []}, decision)
    assert result.status_code == 200
    assert result.model == "model-b"
    assert "model-b" in calls


def test_circuit_breaker_two_tier() -> None:
    """Model-specific errors must not trip the provider circuit breaker."""
    cb = ProviderCircuitBreaker(failure_threshold=3)

    # 5 consecutive model-level errors (404, 400, 429)
    for _ in range(5):
        cb.fail("nim", is_transport=False, model_id="meta/llama-3.3-70b-instruct")

    # Provider breaker MUST still be closed!
    assert cb.state("nim") == BreakerState.CLOSED
    assert cb.allow("nim") is True
    assert cb.is_model_on_cooldown("meta/llama-3.3-70b-instruct") is True

    # But transport failures DO increment failures
    cb.fail("nim", is_transport=True)
    cb.fail("nim", is_transport=True)
    cb.fail("nim", is_transport=True)
    assert cb.state("nim") == BreakerState.OPEN
    assert cb.allow("nim") is False


@pytest.mark.asyncio
async def test_explicit_request_graceful_fallback_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When allow_graceful_fallback_on_explicit is True, explicit requests fall back to available models."""
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=5,
        request_deadline_seconds=10.0,
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
        allow_graceful_fallback_on_explicit=True,
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
    assert result.status_code == 200
    assert result.model == "model-b"
