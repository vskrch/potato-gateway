# Potato Gateway — Priority Remediation Plan (→ 99.99 %)

Companion to `docs/ha-audit-report.md`. Every change below cites the exact defect (D*) and file/line it targets. Phases are ordered so each ships independently behind tests; P0 items are one- to two-day changes.

**Evidence artifact:** `scripts/rca_repro.py` (run: `uv run python scripts/rca_repro.py`) reproduces D1 and D2 today. Promote it to CI (see T0) so the defect can never return.

---

## P0 — Stop producing 503s that recovery was designed to prevent

### P0-1 · Decouple the recovery phases from the primary-chain budget (D1)

**Defect:** `max_attempts == chain budget`, chain is sized to that budget (`fallback.py:1187`), recovery gated on `attempts < max_attempts` (`:1664`, `:1866`, `:2611`, `:3008`) ⇒ never reached.

**Change:** introduce an explicit recovery reserve. The universal ceiling (`max_model_fallbacks`) still bounds total attempts; when the cap is 1 the reserve is 0, preserving `tests/test_fallback.py:747` semantics.

```diff
--- a/src/potato/routing/fallback.py
+++ b/src/potato/routing/fallback.py
@@ execute_json (JSON path, after `from potato.compat import openai_error`)
-        deadline = self._make_deadline(decision.intent.value)
-        max_attempts = max(1, self._max_n_for_intent(decision.intent.value))
-        attempts = 0
+        deadline = self._make_deadline(decision.intent.value)
+        chain_budget = max(1, self._max_n_for_intent(decision.intent.value))
+        # D1: recovery (last-resort + graceful) must not be budget-starved by
+        # the primary chain. Reserve attempts for recovery phases; total stays
+        # bounded by chain_budget + recovery_budget and by the deadline.
+        # cap == 1 → reserve 0 (universal ceiling semantics preserved).
+        recovery_budget = max(0, min(3, chain_budget - 1))
+        max_attempts = chain_budget + recovery_budget
+        attempts = 0
```

Same edit in `execute_stream` (context `last_status = 503 / saw_deadline = False`, `:1975–1983`).

Because `_chain()` still caps the primary chain at `chain_budget`, the primary loop consumes ≤ `chain_budget` attempts; both recovery gates (`attempts < max_attempts`) now admit ≥ 1 (up to 3) recovery attempts, and `_try_models(..., max_attempts=max(0, max_attempts - attempts))` / `untried_any[: max(0, max_attempts - attempts)]` become non-empty.

**Verify with T1.**

---

### P0-2 · A circuit-open *skip* must never count as a transport failure (D2)

**Defect:** `fallback.py:2008–2009` (also `:2656`, `:3038`) call `_circuit_fail(pid)` when `client_for_model` raises — but `hub.py:210–214` raises for skip/disabled/no-keys states with the explicit contract *"don't trip the circuit breaker"*. In HALF_OPEN this re-opens the circuit before the probe resolves (reproduced).

```diff
--- a/src/potato/routing/fallback.py
+++ b/src/potato/routing/fallback.py
@@ execute_stream
             pid = self._provider_id_for(model)
             try:
                 client, upstream_mid = self._client_for(model)
             except RuntimeError as exc:
-                self._circuit_fail(pid)
+                # D2: client_for_model raises for circuit-open SKIP / disabled
+                # runtime / missing keys — none are transport failures. The
+                # breaker already knows it is open; re-failing it here used to
+                # re-open HALF_OPEN probes and cascade across providers.
+                logger.info("stream skip %s: %s; advancing", model, exc)
                 if idx < len(chain) - 1:
                     self.stats.fallback_advances += 1
                     continue
```

Apply the same deletion at `:2656` (fresh phase, `except RuntimeError: self._circuit_fail(pid2); continue`) and `:3038` (graceful phase).

**Hardening (same PR):** make the contract explicit in `catalog/hub.py`:

```python
class ProviderCircuitOpen(RuntimeError): ...
class ProviderNotConfigured(RuntimeError): ...

# client_for_model:
if not self.circuit_breaker.allow(pid):
    ...
    raise ProviderCircuitOpen(f"provider '{pid}' circuit is open ...")
...
if rt is None or not rt.config.enabled:
    raise ProviderNotConfigured(f"provider '{pid}' is not available ...")
raise ProviderNotConfigured(f"provider '{pid}' has no API keys ...")
```

`FallbackExecutor` catches `ProviderCircuitOpen/ProviderNotConfigured` → advance with **no** health/breaker mutation (breaker already open / config already known); plain `RuntimeError` (key-pool exhaustion) continues to reach the *capacity* handling in P0-3.

---

### P0-3 · Enforce two-tier classification at every failure site (D3)

**Defect:** 6 sites push model-tier/rate-limit events into the transport breaker; `cb.model_fail` is dead code.

**3a. One classifier, used everywhere** (new helper on `FallbackExecutor`):

```python
MODEL_TIER_STATUSES = {400, 401, 403, 404, 405, 408, 413, 422, 429}

def _record_circuit(self, provider_id: str | None, *, model: str,
                    status: int | None = None, transport: bool = False) -> None:
    """Two-tier circuit classification (D3).
    transport=True  → provider transport breaker (5xx / timeouts / transport exc)
    status in MODEL_TIER → model-tier cooldown only; provider breaker untouched.
    """
    if transport:
        self._circuit_fail(provider_id)                     # is_transport=True
        return
    if status is not None and status in MODEL_TIER_STATUSES and provider_id and self.hub:
        cb = getattr(self.hub, "circuit_breaker", None)
        if cb is not None:
            cb.fail(provider_id, is_transport=False, model_id=model)
```

**Call-site changes:**

| Site | Today | Becomes |
|---|---|---|
| `_try_models:762` (any non-2xx) | `_circuit_fail(pid)` | `self._record_circuit(pid, model=model, status=status)` |
| `_try_models:699` (2xx empty reply) | `_circuit_fail(pid)` | model tier only: `cb.fail(pid, is_transport=False, model_id=model)` via `_record_circuit(..., status=422)` semantics — **quality, not transport** |
| `:1836` (fresh, any status) | `_circuit_fail(pid2)` | `self._record_circuit(pid2, model=model2, status=s2)` |
| `:1491` | `elif status >= 500: _circuit_fail(pid)` | `elif status >= 500: self._record_circuit(pid, model=model, transport=True)` (behavior preserved, now uniform) |
| `:1383`, `:2055` (TimeoutError) | `_circuit_fail(pid)` | transport **only if bytes were in flight**; capacity timeouts (below) → no transport fail |

**3b. Typed capacity vs. stall inside `upstream.py`** so `wait_for` timeouts stop poisoning the breaker:

```python
class KeyPoolExhausted(RuntimeError): ...   # acquire() failure / all keys cooling
class RetryAfterSleepCancelled(asyncio.CancelledError): ...  # optional

# upstream.request_json / stream:
try:
    key = await self.pool.acquire(preferred_key_id=preferred_key_id)
except RuntimeError as exc:
    raise KeyPoolExhausted(str(exc)) from exc
```

In `fallback.py`:

```python
except KeyPoolExhausted:
    # rate-limit / capacity — model+key tier only (health 429 cooldown already
    # recorded via _record_outcome); never the transport breaker.
    self.stats.fallback_advances += 1
    continue
except TimeoutError:
    # Distinguish: if the pool/Retry-After path was sleeping, it is capacity.
    ...
```

For `TimeoutError`, the cheap correct split: have `UpstreamClient` wrap its internal `sleep_backoff` and `pool.acquire` waits so a `wait_for` cancellation surfaces which phase was active (set `self._phase = "acquire" | "retry_after" | "in_flight"` is unsafe cross-task — instead pass a `budget_phase` marker by structuring the call: `acquire()` **before** entering the timed region is the cleanest fix:

```python
# execute_json / execute_stream attempt sites
key = await asyncio.wait_for(client.pool.acquire(...), timeout=2.0)   # capacity
# then time only the HTTP:
status, body, headers, key = await asyncio.wait_for(
    client.request_json(..., key=key), timeout=attempt_budget)        # stall ⇒ transport
```

(add optional `key=` parameter to `request_json`/`stream`; on capacity failure → `KeyPoolExhausted` path, no transport fail).

**3c. Wire the model tier end-to-end:** `_circuit_succeed/_circuit_fail` gain `model_id=model`, and `_chain()` consults `cb.is_model_on_cooldown(f"{pid}/{mid}")` (provider-namespaced) alongside `ModelHealthStore` — so the breaker's model tier stops being vestigial and provider-scoped model cooldowns survive restarts of health state.

---

### P0-4 · Scope last-resort recovery to the request (D4)

**Defect:** `fallback.py:1665–1670` and `:2612–2617` wipe **all** model cooldowns and `force_allow()` **all** providers per request.

```diff
--- a/src/potato/routing/fallback.py
+++ b/src/potato/routing/fallback.py
@@ execute_json last-resort
             remaining = deadline - time.monotonic()
             if remaining >= 5.0 and attempts < max_attempts:
-                if hasattr(self.registry, "health"):
-                    for h in self.registry.health._by_model.values():
-                        h.cooldown_until = 0.0
-                if self.hub is not None:
-                    for pid in self.hub.provider_ids:
-                        self.hub.circuit_breaker.force_allow(pid)
                 retry_chain = self._chain(decision, had_tools=had_tools)
+                # D4: recovery is SCOPED. Never mutate cooldowns/breakers of
+                # models/providers this request is not about to try.
+                # (Blanket force_allow removed: hub.client_for_model already
+                # force-allows when EVERY active provider is open, and the
+                # breaker's own probe timer handles single-provider recovery.)
+                for m in retry_chain:
+                    h = self.registry.health._by_model.get(m)
+                    if h is not None:
+                        h.cooldown_until = 0.0
                 if not retry_chain:
                     ... (existing heal / emergency_chain unchanged)
```

Mirror in `execute_stream` (`:2611–2618`).

**Escape hatch (optional, if operators want it):** `force_allow(pid, *, min_interval=30.0)` — no-op when a force already happened within the interval:

```diff
--- a/src/potato/safety/circuit_breaker.py
+++ b/src/potato/safety/circuit_breaker.py
@@
-    def force_allow(self, provider_id: str) -> None:
+    def force_allow(self, provider_id: str, *, min_interval: float = 0.0) -> None:
         pid = provider_id.lower()
+        now = time.monotonic()
+        if min_interval and now - self._last_force.get(pid, 0.0) < min_interval:
+            return  # D4: recovery may not defeat the breaker more than once/interval
+        self._last_force[pid] = now
         self._state[pid] = BreakerState.HALF_OPEN
         self._failures[pid] = 0
         self._open_until.pop(pid, None)
```
(add `self._last_force: dict[str, float] = {}` to `__init__`.)

---

### P0-5 · Bound every sleep by the request deadline (D5b)

```diff
--- a/src/potato/safety/backoff.py
+++ b/src/potato/safety/backoff.py
@@
-def compute_backoff_seconds(attempt: int, *, base=0.5, cap=16.0,
-                             retry_after: float | None = None) -> float:
+def compute_backoff_seconds(attempt: int, *, base=0.5, cap=16.0,
+                             retry_after: float | None = None,
+                             max_delay: float | None = None) -> float:
     exp = min(cap, base * (2 ** max(0, attempt)))
     delay = exp
     if retry_after is not None and retry_after > 0:
         delay = max(delay, float(retry_after))
     delay *= 1.0 + random.uniform(0.0, 0.2)
+    if max_delay is not None:          # D5b: never sleep past the deadline
+        delay = min(delay, max(0.0, max_delay))
     return max(0.0, delay)

 async def sleep_backoff(attempt: int, *, base=0.5, cap=16.0,
-                         retry_after: float | None = None) -> float:
-    delay = compute_backoff_seconds(attempt, base=base, cap=cap, retry_after=retry_after)
+                         retry_after: float | None = None,
+                         max_delay: float | None = None) -> float:
+    delay = compute_backoff_seconds(attempt, base=base, cap=cap,
+                                     retry_after=retry_after, max_delay=max_delay)
```

Every in-loop call site passes `max_delay=remaining - 2.0` (JSON `:1396,1450,1621,1629,1647`; stream `:2068,2093,2239,2271,2563,2571`), and `upstream.py` passes `max_delay=attempt_budget_remaining`. Additionally, for **429 with other candidates available, do not sleep at all** — set the model/key cooldown and advance immediately (sleep only when the rate-limited candidate is the last viable one).

---

### P0-6 · User rate limit must return 429, not 500 (D5c)

```diff
--- a/src/potato/routes/openai.py
+++ b/src/potato/routes/openai.py
@@ _prepare_routed
     try:
         ctx = await guard.before_request(headers=request.headers, proxy_token=proxy_token, body=body)
     except RateLimitedError as exc:
-        return JSONResponse(content=exc.response, status_code=429)
+        raise  # D5c: caller (_chat_like:707 / embeddings:1215) maps this to 429;
+               # returning a JSONResponse here made the caller unpack a 5-tuple
+               # from a Response → TypeError → 500.
```

---

### P0-7 · Make "fresh" actually fresh + honest terminal errors (D5a, D5d)

```diff
--- a/src/potato/routing/fallback.py
+++ b/src/potato/routing/fallback.py
@@ execute_json last-resort fresh selection
-                retry_chain = self._chain(decision, had_tools=had_tools)
-                ...
-                tried = {m.lower() for m in chain}
-                fresh = [m for m in (retry_chain or []) if m.lower() not in tried]
+                tried = {m.lower() for m in chain}
+                # D5a: last-resort must draw from the WIDEST pool, not rebuild
+                # the same deterministic chain (fresh would be ~empty).
+                pools: list[list[str]] = []
+                try:
+                    pools.append(self._any_available_live_models(
+                        had_tools=had_tools, intent=decision.intent.value,
+                        allowed_models=list(getattr(decision, "allowed_models", None) or []),
+                        free_only=(str(getattr(decision, "auto_tier", "") or "").lower() == "free"),
+                    ))
+                except Exception:
+                    logger.debug("recovery pool build failed", exc_info=True)
+                pools.append(self._chain(decision, had_tools=had_tools) or [])
+                seen, fresh = set(tried), []
+                for pool in pools:
+                    for m in pool:
+                        if m.lower() not in seen:
+                            seen.add(m.lower())
+                            fresh.append(m)
+                if not fresh and not pools[1]:
+                    ... (existing heal / emergency_chain unchanged)
```

Mirror for the stream path (`:2618–2636`).

Terminal envelope (`:1904–1921`) — stop leaking provider text as gateway text, add backoff hint:

```python
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
            # raw body capped + moved out of `message` (D5d)
            "upstream_detail": str(last.body)[:300],
        },
    ),
    headers={**last.headers, "Retry-After": "3"},
    ...
)
```

**Status-code policy (ship with the same PR):** gateway-generated failures split into `502 all_providers_failed` (upstream fault), `503 capacity_exhausted` (+`Retry-After`), `504 request_deadline_exceeded`. Keep 503 only where a retry window is meaningful.

---

## P1 — Streaming correctness, socket lifecycle, deadline hygiene

### P1-1 · Mid-stream upstream error-frame reconciliation (D6a)

In `robust_iter._scan_for_tokens` → rename to `_scan_frame` and add (after the line is a complete `data:` payload):

```python
def _scan_frame(self, c: bytes) -> None:
    for raw_line in c.split(b"\n"):
        if not raw_line.startswith(b"data:"):
            continue
        payload = raw_line[5:].strip()
        if payload in (b"[DONE]", b"") or not payload.startswith(b"{"):
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
            nonlocal upstream_error
            upstream_error = obj["error"]      # → stream_failed + model tier
        # existing usage extraction stays here
```

On `upstream_error`: set `held.stream_failed = True`, `cb.fail(pid, is_transport=False, model_id=mid)` (mid-stream provider error = model tier; use `transport=True` only if the socket died), `_record_outcome(success=False, status_code=502)`, and re-emit the error as a **clean gateway error frame** (`code=upstream_stream_error`) instead of relaying the provider's raw frame — then `[DONE]`. Health store then learns; clients see a canonical envelope; request logs stop counting these as successes.

### P1-2 · Deterministic async-generator cleanup + cancel-safe socket close (D6b)

Add explicit close propagation at every layer:

```python
# compat.normalize_sse_stream
async def normalize_sse_stream(source, *, routed_model=None):
    buffer = b""
    try:
        ...existing loop...
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()

# fallback.robust_iter finally:
finally:
    _cancel_producer()
    await _await_producer()
    aclose = getattr(rest, "aclose", None)          # NEW: close upstream iterator
    if aclose is not None:
        with suppress(Exception):
            await aclose()

# routes/openai._gated_stream finally:  (before guard.after_request)
with suppress(Exception):
    await upstream_iter.aclose()
```

`upstream.stream()` cancel window:

```diff
             except BaseException as exc:
-                if not released:
-                    await self.pool.release(key, success=False)
+                # D6b: never leak the streamed response on cancellation
+                if "resp" in locals() and resp is not None:
+                    with suppress(Exception):
+                        await resp.aclose()
+                if not released:
+                    await self.pool.release(key, success=False)
                 if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                     raise
```

### P1-3 · Fail-fast HTTP timeouts (D6c)

```diff
--- a/src/potato/upstream.py
+++ b/src/potato/upstream.py
-            "timeout": httpx.Timeout(self.timeout, connect=self.connect_timeout),
+            "timeout": httpx.Timeout(
+                self.timeout,                       # read (full generation)
+                connect=self.connect_timeout,       # 5s
+                pool=getattr(self.settings-ish, "upstream_pool_timeout", 10.0),
+                write=30.0,
+            ),
```
(config: `upstream_pool_timeout: float = 10.0` — a saturated 200-conn pool must fail in seconds, not 5 minutes). Route the **passthrough** (`routing disabled`) path through the same `wait_for(request_deadline)` wrapper as routed traffic.

### P1-4 · Deadline decomposition (applies to P0-1)

Replace ad-hoc `remaining < 1.0 / 3.0 / 5.0` gates with:

```python
def _phase_deadlines(deadline, now):
    total = deadline - now
    return {
        "primary":  now + total * 0.60,
        "recovery": deadline - total * 0.15,   # recovery runs until 85%
        "guard":    2.0,                        # structured error + logging
    }
```
Primary loop checks `now < d["primary"]`; recovery phases `now < d["recovery"]`; never sleep past `deadline - guard`.

### P1-5 · Regression suite (ships with P0/P1)

| Test | Asserts |
|---|---|
| `test_recovery_runs_when_chain_sized_to_budget` (T1) | with `intent_max_fallbacks=6` and ≥6 failing chain models + 1 healthy untried model → **200**, recovery phases entered |
| `test_stream_circuit_skip_does_not_trip_breaker` (T2) | breaker HALF_OPEN, chain has 2 models on that provider → after skips, breaker **not** re-opened by skips |
| `test_model_errors_do_not_open_provider_breaker` (T3) | `_try_models` over 400/404/429/empty-2xx → provider `state == CLOSED`, models on cooldown |
| `test_last_resort_is_scoped` (T4) | unrelated model's cooldown survives; unrelated provider stays OPEN after recovery |
| `test_retry_after_sleep_bounded_by_deadline` (T5) | `Retry-After: 3600` with 10 s deadline → returns within ~10 s |
| `test_user_rate_limit_returns_429` (T6) | `user_rpm_limit=1` → second request 429, body `user_rate_limited` |
| `test_fresh_phase_draws_from_any_live_pool` (T7) | `fresh` contains untried models when `_chain()` is deterministic |
| `test_terminal_503_has_retry_after_and_no_raw_last_body` (T8) | headers + envelope shape |
| `test_midstream_provider_error_frame_marks_failure` (T9) | `stream_failed`, health error_count+, gateway error frame |
| `test_generator_close_releases_key_inflight` (T10) | consumer abandons stream mid-flight → `key.in_flight == 0` without GC |
| Promote `scripts/rca_repro.py` → `tests/test_rca_recovery_budget.py` | fails on D1 regression |

Update `tests/test_fallback.py:747` docstring: ceiling = `chain_budget + recovery_budget`, reserve 0 when cap == 1 (assert unchanged).

---

## P2 — Architecture for the P99 / 99.99 % target

### P2-1 · True TTFT speculative hedging (replaces sequential fail-fast)

Design (flag: `enable_parallel_hedge`, default off → canary → on):

```python
async def open_with_hedge(candidates: list[Candidate], *, budget: float,
                          hedge_delay: float, max_hedges: int = 1):
    """Race primary now, secondary after hedge_delay; first content-bearing
    delta wins; losers cancelled. One hedge max, global hedge budget enforced."""
    tasks = [(c, asyncio.create_task(open_stream(c))) for c in candidates[:2]]
    # winner = first task to produce a content-bearing chunk within budget
    ...
    for c, t in tasks:
        if t is not winner:
            t.cancel()
            # record loser TTFT + any billed usage (include_usage probe)
```

- Trigger hedge only when: `ewma_ttft(p95) > hedge_threshold`, or primary is a retry, or breaker is half-open — never for every request.
- Budgets: per-request `max_hedges=1`, global `max_concurrent_hedges = 5% of gate capacity`; hedge candidates must be **provider-diverse**.
- Apply uniformly to chain and recovery phases (fixes the flat-15 s inconsistency at `:2744`/`:3125`).
- Cost control: record abandoned-attempt usage when the provider reports it; expose `hedge_abandoned_tokens` metric.
- Expected: P99 TTFB `T1+T2 → min(T1, hedge_delay+T2)`; with `hedge_delay ≈ P95 TTFT(primary)`.

### P2-2 · Tier-annotated adaptive fallback matrix (D7b)

`_chain()` returns `list[Candidate(model, provider, tier, reason)]` per the T0–T5 tiers in the audit, with these matrix rules keyed by **error class** (LiteLLM-style):

| Error observed | Next action |
|---|---|
| 429 (model/key) | same model, other provider (T1) — **no sleep** |
| 429 (provider capacity) | provider's `rate-limit budget` → skip provider entirely this request (T2+) |
| 400 context-overflow | same provider, reduced `max_tokens` retry ×1, then T2 |
| 400 tools-unsupported | capability-filtered tier (learned caps already exist) |
| 5xx / transport / TTFT-stall | next provider first (diversity), then T2/T3 |
| model-404 | horizontal siblings (T1), then T2 |
| all above exhausted | T4 emergency frontier (≥1 healthy strong model, allowlist-respecting) → T5 structured error |

**Provider-diversity constraint (critical):** no more than 2 consecutive same-provider candidates; guarantee ≥2 distinct providers in the first `min(len, 4)` candidates whenever available. This converts single-provider outages from "chain consumed → 503 (D1)" into "advance one slot → serve", the single biggest correlated-failure reducer for the 53-ppm budget.

### P2-3 · Breaker & health state for multi-replica HA (D7c)

- Windowed rates: open when `failures ≥ 5 AND failure_rate ≥ 0.5 in 30 s` (Envoy outlier-detection semantics) instead of consecutive counts; decay `_failures` on success.
- Half-open: probe **token** acquired at attempt start, released on outcome (`probe_in_flight`), instead of `_last_probe` timestamp guess.
- Split `rate-limit budget` from transport breaker (P0-3c).
- Replica-local state is acceptable if sticky routing is consistent (note: `StickySessionStore` is per-process too — pin sessions by `X-Potato-Key-Id` hashing or move stickiness to the LB); otherwise promote breaker/health to shared Redis with 1 s TTLs. Alert when two replicas disagree on a provider's state > 60 s.

### P2-4 · Observability & SLO instrumentation (required to *prove* 99.99 %)

Metrics (labels: `provider`, `tier`, `phase`, `intent`):

- `potato_requests_total{status_class}` / `potato_5xx_total{code}` — the SLO numerator
- `potato_fallback_advance_total{phase}` — phase = `primary|last_resort|graceful|hedge`
- `potato_recovery_phase_entered_total` and `potato_chain_exhausted_total` (D1's smoking gun: should be ~0 once P0-1 ships)
- `potato_breaker_transitions{provider,from,to}` + `potato_breaker_probe_outcomes`
- `potato_ttft_seconds` histogram (p50/p95/p99 by provider) + `potato_hedge_*
- `potato_deadline_exceeded_total{phase}` and `potato_attempt_sleep_seconds_total`
- `potato_gate_in_flight` vs `potato_key_in_flight` (leak detector for D6b)
- Error-budget burn-rate alerts (fast burn 14.4× over 5 min page).

---

## Rollout plan

| Phase | Contents | Effort | Risk | Gate |
|---|---|---|---|---|
| **0** | T1–T10 tests + `rca_repro` in CI + metrics from P2-4 (counters first) | 0.5 d | none | CI green, dashboards show current `chain_exhausted` rate |
| **1 (P0)** | P0-1 … P0-7 | 1–2 d | low — each behind its test; P0-1 changes attempt counts only | `potato_5xx_total` − expected ≥80 % on the 503 class; `chain_exhausted` ≈ 0 |
| **2 (P1)** | P1-1 … P1-5 | 2 d | low | stream `stream_failed` detection live; no in_flight drift over 24 h |
| **3 (P2)** | Matrix + diversity (P2-2), windowed breakers (P2-3) | 3–4 d | medium — routing order changes; ship behind `enable_tiered_matrix` canary 10 % → 100 % | advance-rate ↓, per-request attempts ↓, quality (RL reward) not degraded |
| **4 (P2)** | Parallel hedging (P2-1) | 3 d | medium — token cost ↑ ~2–5 %; gate with `enable_parallel_hedge` + hedge budget | P99 TTFB ↓ ≥ 30 %, `hedge_abandoned_tokens` within budget |
| **5** | Multi-replica state, chaos drills (kill provider / 429 storms / slow-loris read stalls) + synthetic probe every 30 s (existing `catalog/prober.py`, raise `ProbeBudget` for synthetic lane) | 2 d | — | game-day: injected single-provider outage produces 0 gateway 5xx for 15 min |

## Path to 99.99 % (error-budget model)

- 99.99 % = **52.6 min/yr · 60.5 s/wk · ≈53 gateway-5xx per million requests** (4xx excluded).
- With P0–P1 shipped, per-attempt independent failure `p` (from `fallback_advance` ratio, typically 1–3 % degraded) across `k = 4` **provider-diverse** candidates ⇒ residual ≈ `p^k` ≈ `10⁻⁷…10⁻⁶` — inside budget with headroom.
- The remaining term is **correlated** failure (one provider = several chain slots). P2-2's diversity constraint + T4 emergency frontier is what takes correlated risk from `P(provider outage)` to `P(all providers down)`; N+1 provider redundancy is a hard prerequisite (alert if eligible providers < 2).
- Deadlines (P1-4) convert pathological stalls into fast 504s *before* the client gives up, keeping tail failures out of the SLO numerator when a fallback can still serve.
- Finally: the generic `except Exception → 503` at `routes/openai.py:1140–1163` must be narrowed (log + `500 internal` for programmer errors, keep the friendly 503 only for `potato_pool_exhausted`/known states) — masking real bugs as retriable 503s is how HA regressions ship quietly.
