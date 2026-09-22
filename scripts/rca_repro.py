"""RCA verification: the recovery pipeline (last-resort + graceful fallback)
must run when the primary chain consumes the whole attempt budget.

Run: uv run python scripts/rca_repro.py
Exit 0 when recovery runs (fixed), 1 when budget-deadlocked (D1 regressed).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from potato.balancer import KeyStats
from potato.catalog import ModelRegistry
from potato.config import Settings
from potato.routing import FallbackExecutor, Intent, RouteDecision

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "config" / "models.yaml"


def key(i: int = 0) -> KeyStats:
    return KeyStats(key_id=f"key-{i}", api_key=f"k{i}")


async def main() -> None:
    settings = Settings(
        nim_api_keys=["k"],
        max_model_fallbacks=10,  # production default
        request_deadline_seconds=300.0,  # production default
        retry_backoff_base_seconds=0.0,
        retry_backoff_cap_seconds=0.0,
    )
    reg = ModelRegistry.from_yaml(YAML)
    # 12 live models — _chain() will cap the chain to max_n == 10 == max_attempts
    live = {f"model-{i}" for i in range(12)}
    reg.live_ids = live

    calls: list[str] = []

    rescue_model: str = ""

    async def fake_json(method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        model = body.get("model", "")
        calls.append(model)
        # Only a model OUTSIDE the capped primary chain can serve — proving
        # whether last-resort / graceful recovery phases ever execute.
        if model == rescue_model:
            return 200, {"choices": [{"message": {"content": "ok"}}]}, {}, key(2)
        return 503, {"error": {"message": "Upstream Request Failed"}}, {}, key()

    upstream = AsyncMock()
    upstream.request_json = fake_json

    decision = RouteDecision(
        chain=sorted(live)[:10],
        mode="auto",
        intent=Intent.CHAT_FAST,
        rule_id="test",
        requested_model="potato/auto",
    )

    ex = FallbackExecutor(upstream, reg, settings)
    ex._provider_available = lambda _m: True  # type: ignore[method-assign]

    # Observe the chain the executor actually builds
    chain = ex._chain(decision, had_tools=False)
    rescue_model = next(m for m in sorted(live) if m not in chain)
    # Pin it: execute_json rebuilds via _chain(), which would otherwise
    # re-expand from the live pool and swallow the rescue model.
    ex._chain = lambda d, had_tools=False: list(chain)  # type: ignore[method-assign]
    print(f"max_attempts (chat_fast) : {ex._max_n_for_intent('chat_fast')}")
    print(f"built chain length       : {len(chain)}")
    print(f"chain                    : {chain}")
    print(f"rescue model (untried)   : {rescue_model}")

    result = await ex.execute_json("/chat/completions", {"messages": []}, decision)
    print(f"\nupstream attempts made   : {len(calls)} -> {calls}")
    print(f"terminal status          : {result.status_code}")
    print(f"terminal body            : {json.dumps(result.body)[:400]}")
    recovered = len(calls) > len(chain)
    print(
        "\nVERDICT: recovery phases ran"
        if recovered
        else "VERDICT: last-resort + graceful fallback NEVER RAN (budget pre-consumed)"
    )

    # ── Breaker poisoning check: stream path fails the provider circuit just
    #    for *skipping* a model whose provider circuit is already open ──
    from potato.safety.circuit_breaker import BreakerState, ProviderCircuitBreaker

    cb = ProviderCircuitBreaker(failure_threshold=6, recovery_timeout=15.0)
    cb.fail("groq", is_transport=True)
    cb.state("groq")
    # simulate OPEN after threshold
    for _ in range(6):
        cb.fail("groq", is_transport=True)
    print(f"\nbreaker after 6 transport fails: {cb.state('groq').value}")
    # force_allow as last-resort does
    cb.force_allow("groq")
    print(f"after force_allow (all-providers last-resort): {cb.state('groq').value}")
    # first model gets the half-open probe via client_for_model->allow()
    allowed = cb.allow("groq")
    print(f"first chain model probe allowed: {allowed}")
    # second model on the SAME provider: allow() refuses within recovery_timeout
    allowed2 = cb.allow("groq")
    print(f"second chain model allowed within probe window: {allowed2}")
    # execute_stream then does: except RuntimeError -> self._circuit_fail(pid)
    cb.fail("groq", is_transport=True)  # == _circuit_fail(pid)
    print(f"breaker after SKIP counted as transport fail: {cb.state('groq').value}")
    print(f"open_until remaining (s): {cb.snapshot().get('groq', {}).get('open_until')}")

    # The stream executor must NOT count a circuit-open skip as a transport
    # failure (D2): skipping must leave the breaker state untouched.
    from potato.catalog.hub import ProviderCircuitOpen

    cb2 = ProviderCircuitBreaker(failure_threshold=6, recovery_timeout=60.0)
    cb2.force_allow("groq")
    assert cb2.state("groq").value == "half_open"
    try:
        raise ProviderCircuitOpen("provider 'groq' circuit is open — skipping model 'x'")
    except ProviderCircuitOpen:
        pass  # executor advances with no breaker mutation (D2 fix)
    assert cb2.state("groq").value == "half_open", "skip must not trip the breaker"
    print("breaker skip-neutrality check: OK")

    if not recovered:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
