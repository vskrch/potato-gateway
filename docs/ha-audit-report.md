# Potato Gateway — High-Availability Architectural & Code Audit

**Scope:** Persistent HTTP 503 "Upstream Request Failed" errors; failover pipeline; production hardening vs. LiteLLM / OpenRouter / RouteLLM / Envoy AI Filters.
**Primary artifacts audited:** `routing/fallback.py` (3,348 L), `safety/circuit_breaker.py`, `routing/selector.py`, `upstream.py`, plus `catalog/hub.py`, `catalog/health.py`, `balancer.py`, `safety/guard.py`, `routes/openai.py`, `compat.py`.
**Method:** Full code-path review of both execution pipelines (`execute_json`, `execute_stream`), state-machine analysis of the breaker/key-pool/health triad, and an executable reproduction (`scripts/rca_repro.py`).
**Baseline:** `pytest -q` → **498 passed**. The suite is green *because it encodes the defective budget behavior* (see D1).

---

## 1. Executive summary

The gateway already has **more resilience machinery than most production gateways**: an ordered fallback chain, retryable-class error taxonomy, per-model health cooldowns with adaptive growth, TTFT/idle watchdogs, mid-stream error framing, cancel-safe key release, a self-heal loop, and a two-tier breaker *API*. The 503s are **not caused by missing machinery — they are caused by five control-plane defects that disable that machinery exactly when it is needed**:

| # | Defect | Effect in production |
|---|--------|----------------------|
| **D1** | Primary-chain attempt budget == total attempt budget; the chain is sized to that same budget ⇒ **last-resort + graceful recovery phases are unreachable** whenever every chain model got a real HTTP attempt. | `503 potato_models_exhausted` while healthy, never-tried models exist. **Reproduced.** |
| **D2** | Stream path counts a *circuit-open skip* as a **transport failure** (`_circuit_fail` on `RuntimeError` from `client_for_model`). | One half-open probe → other models on that provider are skipped → skips re-open the breaker → **breaker cascade across providers**. |
| **D3** | Two-tier discipline is violated at 6 call sites: model-tier errors (400/404/429/empty-reply) and rate-limit-induced timeouts trip the **provider transport** breaker; the model-tier breaker API (`model_fail`/`is_model_on_cooldown`) is **dead code**. | Healthy providers opened on model errors; model-level cooling lives only in `ModelHealthStore`. |
| **D4** | Last-resort recovery performs a **global** wipe of every model cooldown + `force_allow()` on **every** provider, per request. | Under sustained incident, breakers/health are repeatedly reset by the requests they are protecting → death-sprints against dead endpoints → deadline burn → 503. |
| **D5** | "Fresh model" recovery re-runs the *same* `_chain()` builder and diffs against `tried` ⇒ `fresh ≈ ∅`; `Retry-After` sleeps are unbounded; user-rate-limit path returns a `JSONResponse` that the caller unpacks as a 5-tuple → **500**. | Recovery phases are no-ops even when budget remains; deadline overruns; 500s when per-user limits are enabled. |

**Root cause of the exact symptom** (`503`, `error.code = potato_models_exhausted`, `last_body.message = "Upstream Request Failed"`): an upstream provider wraps its own failure in `400/5xx` with the message *"Upstream Request Failed"* → `_is_retryable_model_error()` correctly advances the chain → the **entire chain is consumed as budget** → both recovery phases are gated on `attempts < max_attempts` and skip → the 503 envelope embeds the provider's `last_body` verbatim (and carries **no `Retry-After`**).

Reproduction (`uv run python scripts/rca_repro.py`, production defaults `max_model_fallbacks=10`, `chat_fast` cap 6):

```
max_attempts (chat_fast) : 6
built chain length       : 6
rescue model (untried)   : model-4
upstream attempts made   : 6 -> ['model-0','model-2','model-1','model-10','model-11','model-3']
terminal status          : 503
terminal body            : {"error":{"message":"All models in routing chain failed.",
                             "code":"potato_models_exhausted","last_status":503,
                             "last_body":{"error":{"message":"Upstream Request Failed"}}}}
VERDICT: last-resort + graceful fallback NEVER RAN (budget pre-consumed)

breaker: force_allow → half_open → 1st model probe allowed → 2nd model DENIED
         → stream path _circuit_fail(skip) → breaker OPEN again (poisoned)
```

Availability implication: each defect converts *independent* upstream failures into *correlated* gateway failures. Layered correctly, per-attempt failure `p≈1%` across 4 provider-diverse attempts yields `p⁴ = 1e-8` — comfortably inside the 99.99 % (≈53 ppm) budget. As shipped, correlated breaker poisoning + budget deadlock make `P(503 | any provider degraded)` approach `P(top providers degraded)`, which cannot meet 53 ppm.

---

## 2. Request-path anatomy (as built)

```
route (_chat_like)
 ├─ _prepare_routed: classify → guard.before_request (global gate) → selector.resolve → decision.chain
 ├─ FallbackExecutor._chain(decision)          # expand → filter → re-rank → cap to max_n
 ├─ execute_stream / execute_json
 │   ├─ Phase 1 "chain":      for model in chain: attempt (wait_for(intent budget))
 │   │       retryable → sleep_backoff(+Retry-After) → next model
 │   ├─ Phase 2 "last-resort": wipe ALL cooldowns, force_allow ALL providers,
 │   │       retry_chain = _chain() again, fresh = retry_chain − tried
 │   ├─ Phase 3 "graceful":   any live model on any provider (auto decisions only)
 │   └─ terminal: 503 `potato_models_exhausted` / 504 SSE error frame
 └─ response: JSON or StreamingResponse(robust_iter → normalize_sse_stream → _gated_stream)
```

Budget plumbing: `_max_n_for_intent()` sizes **both** the chain (`_chain`, `fallback.py:1187`) and `max_attempts` (`fallback.py:1277` / `:1982`). Phases 2–3 are gated on `attempts < max_attempts` (`:1664`, `:1866`, `:2611`, `:3008`).

---

## 3. Root-cause analysis (RCA)

### R1 — Budget deadlock in the recovery pipeline *(defect D1, P0)*

- `_chain()` caps the executed chain at `max_n` — `fallback.py:1187`: `chain = available[: max(1, max_n)]`.
- `max_attempts = max(1, self._max_n_for_intent(...))` — `fallback.py:1277` (JSON), `:1982` (stream) — **the same `max_n`**.
- Every chain model that returns HTTP consumes `attempts` (`:1484`); transport failures/timeouts also consume it (`:1382`, `:1421`).
- Phase 2 gate: `if remaining >= 5.0 and attempts < max_attempts:` (`:1664`) → `6 < 6` is **False**.
- Phase 3 gate: `if remaining >= 3.0 and attempts < max_attempts:` (`:1866`) → also False; even if it passed, `_try_models(..., max_attempts=max(0, max_attempts - attempts))` = 0 → instant `break` (`:632`).
- Stream path: identical gates (`:2611`, `:3008`) plus `untried_any = untried_any[: max(0, max_attempts - attempts)]` (`:3020`) → empty slice.

Because production catalogs always supply ≥ `max_n` eligible models, **chain length == max_attempts** and both recovery phases are dead code in the exact scenario they exist for. The regression test `tests/test_fallback.py:747` (`test_fallback_honors_universal_attempt_budget`) *asserts* this behavior with `max_model_fallbacks=1`, and all recovery tests use `max_model_fallbacks=5` with chains of 1–2 — so the suite cannot catch it.

**Why it reads as "failovers don't trigger":** classification, health, and chain logic all work (the chain *is* advanced through all 6–10 models); the failure surfaces only at the end, when the recovery layers that would have saved the request are budget-blocked.

### R2 — Breaker poisoning via skip-as-failure *(defect D2, P0)*

- `hub.client_for_model()` (`hub.py:194–214`) **mutates** breaker state through `allow()` (OPEN→HALF_OPEN probe consumption) and raises `RuntimeError` for three *distinct* reasons: circuit-open skip, provider disabled, provider without keys — with comments explicitly stating config states "don't trip the circuit breaker".
- Stream path (`fallback.py:2008–2009`) does, unconditionally:
  ```python
  except RuntimeError as exc:
      self._circuit_fail(pid)      # ← a SKIP is recorded as a transport failure
  ```
  Same at `:2656` (fresh phase) and `:3038` (graceful phase).
- JSON path (`:1296–1302`) does **not** — the two pipelines disagree.
- `cb.fail()` in `HALF_OPEN` immediately re-opens with backoff (`circuit_breaker.py:129–142`). Sequence (reproduced): `force_allow → half_open →` model #1 takes the single probe slot → models #2..N on the same provider are refused within `recovery_timeout` (15 s) → each refusal re-fails the provider → **circuit re-opens before the probe result returns**. With multi-model chains this cascades: provider A's breakers flap, its candidates are dropped by the `_chain` circuit filter (`fallback.py:979–985`), and the whole chain skews onto fewer providers — then R4/R1 finish the job.

Secondary: `allow()` is not idempotent (`blocked()` exists precisely because `allow()` "consumes probe slots", `circuit_breaker.py:77–86`), yet `client_for_model` calls it once **per candidate**, so a 6-model chain can burn 6 probe decisions.

### R3 — Two-tier discipline violations & dead model tier *(defect D3, P0)*

The breaker API supports tiers (`fail(..., is_transport=False, model_id=...)`), and `tests/test_resilience_regression.py:77` asserts the invariant. In `src/`:

1. **`model_fail` / `is_model_on_cooldown` are never reached** — `grep is_transport=False` matches only tests. `_circuit_fail()` is always called with defaults (`is_transport=True`, `model_id=None`), so `cb._model_cooldowns` is vestigial; model cooling happens *only* in `ModelHealthStore`.
2. **Model-tier errors open the transport circuit:**
   - `_try_models:762` — `_circuit_fail(pid)` after **any** non-2xx (400/404/413/422/429 included);
   - `_try_models:699` — 2xx with empty reply → transport fail;
   - `:1836` (fresh phase) — any status → transport fail;
   - Main loop is correct (`:1490` `elif status >= 500`) — another JSON/stream/phase inconsistency.
3. **Rate-limit conditions are classified as transport:**
   - `except TimeoutError → _circuit_fail(pid)` (`:1383`, `:2055`) — but the timeout is `wait_for(request_json/stream, intent_budget)`; the inner coroutine may be sleeping in `KeyPool.acquire` (up to 30 s, `balancer.py:139–190`) or in `sleep_backoff(retry_after=…)` (`upstream.py:162–175`). A 429 storm therefore **opens the transport breaker**, which then skips healthy models via the `_chain` circuit filter.
   - Pool exhaustion `RuntimeError("All NIM API keys are rate-limited…")` also lands in `_circuit_fail`.
4. **No decay/window:** `_failures` is a raw consecutive counter with no time window or success-rate denominator (Envoy outlier detection and LiteLLM cooldown both use windows). Mixed traffic (1 transport failure + 5 model failures misclassified) opens faster than the documented model says.

### R4 — Recovery mutates global state *(defect D4, P0)*

`fallback.py:1665–1670` (and `:2612–2617` stream):

```python
for h in self.registry.health._by_model.values():
    h.cooldown_until = 0.0                      # wipes EVERY model's cooldown
for pid in self.hub.provider_ids:
    self.hub.circuit_breaker.force_allow(pid)   # force-allows EVERY provider
```

- A single request's failure path clears **all** cooldowns — including 429/unavailable cooldowns for models it never touched — for all concurrent requests (state is process-global).
- `force_allow` resets `_failures=0`, so subsequent failures must re-count from zero → under sustained outage every request that reaches phase 2 **re-opens all breakers**, each burns its full attempt budget on dead endpoints, and the resulting slow failures push later requests past their deadlines → more 503s. The breaker is effectively *disabled during incidents*, which is the only time it matters.
- Combined with R1 this runs on *some* requests (those with spare budget), so breaker state flaps non-deterministically — the classic "state poisoning" signature.

### R5 — Recovery that cannot recover *(defect D5, P1)*

- **Fresh phase reuses the same builder:** `retry_chain = self._chain(decision, ...)` (`:1671`, `:2618`) then `fresh = [m for m in retry_chain if m not in tried]` (`:1688–1689`). Deterministic builder + identical inputs ⇒ `fresh ≈ ∅` (ordering changes only via the cooldown wipe of R4). The "last-resort fresh models" phase is, in practice, a no-op.
- **Unbounded `Retry-After` sleeps:** `sleep_backoff(..., retry_after=ra)` makes `delay = max(exp, retry_after)` with no deadline clamp (`backoff.py:24–25`) — called at `:1622`, `:1647`, `:2564`. A `Retry-After: 3600` (or a broken provider sending one) sleeps the request far past its deadline while the global gate slot is still held; combined with the in-loop placement, chain advancement stalls for the sleep duration.
- **User rate-limit → 500:** `_prepare_routed` catches `RateLimitedError` and *returns* a `JSONResponse` (`routes/openai.py:367–370`), but the caller unpacks a 5-tuple (`:700`) → `TypeError` → generic `raise` (`:751`) → app-wide handler → **500**. The correct handler at `:707` is unreachable.
- **Missing `Retry-After` on terminal 503:** `:1904–1921` builds `potato_models_exhausted` with `headers=last.headers` — no `Retry-After`, no tier/breaker telemetry; `last_body` is passed through raw, which is how provider text like *"Upstream Request Failed"* reaches clients under a gateway 503.

### R6 — Streaming: reconciliation & socket lifecycle gaps *(P1)*

**What is good:** `robust_iter` never sends a bare `[DONE]` on failure — it emits `finish_reason:"error"` + an error event + `[DONE]`, sets `stream_failed`, uses a bounded 32-chunk backpressure queue, cancels the producer on idle timeout, and releases the key in `byte_iter.finally`. That is **OpenRouter parity**.

Gaps:

1. **No upstream error-frame detection.** `_scan_for_tokens` (`:2347`) only looks for usage. A provider that returns `200 SSE` then `data: {"error": {...}}` mid-stream passes through untouched; the producer ends *normally* → `_record_outcome(success=True)` (if usage seen) → **breaker and health never learn**, client stream is corrupted, request logged as success.
2. **Commit-on-first-chunk.** The first byte (often a role-only delta or `: ping`) commits the response; everything after is unrecoverable. Failover decisions must be made *before* commit (commit threshold), otherwise a slow-first-token failover can only ever happen before any byte — today's watchdog partially compensates but wastes the attempt's billed tokens.
3. **Cleanup depends on GC finalizers.** The chain is `_gated_stream → normalize_sse_stream → robust_iter → producer → byte_iter`, and **no layer explicitly `aclose()`s its source** on early exit:
   - `normalize_sse_stream` (`compat.py:361`) has no `finally: await source.aclose()`;
   - `robust_iter.finally` cancels the producer but does not `await rest.aclose()`;
   - `routes/openai.py:_gated_stream` has no `finally: await upstream_iter.aclose()`.
   Key release (`in_flight`, sticky, health) rides on CPython async-generator finalization hooks. It usually works; under reference cycles or loop shutdown it leaks slots until GC — and a leaked `in_flight` shrinks effective key capacity permanently (`max_in_flight_per_key=3`).
4. **`upstream.stream()` cancel window:** `except BaseException` (`upstream.py:363–372`) releases the key but never `resp.aclose()`s. Cancellation landing between `client.send(..., stream=True)` completion and iterator handoff (or inside the `pool.release` await) leaks an open upstream connection/socket.
5. **Timeouts:** `httpx.Timeout(self.timeout, connect=5)` (`upstream.py:62`) sets **read/pool/write = 300 s**. `pool_timeout=300` means a saturated connection pool (200 conns) stalls a request up to 5 minutes instead of failing fast; no write/keepalive differentiation.
6. **Passthrough (routing disabled) has no deadline wrapper at all** — `routes/openai.py:1085` calls `upstream.request_json` directly: 3 internal retries + unbounded `Retry-After` sleeps + 300 s read timeout, no fallback by design but also no deadline enforcement.

### R7 — No true hedging; adaptive matrix is implicit *(P2, architectural)*

- `enable_ttft_hedging` is **sequential fail-fast**, not hedging: `effective_ttft = min(ttft, max(3.5, ewma*1.8))` (`:2162–2167`) then wait, close, try next. P99 = `T1_first_token + T2_first_token`; the failed attempt's billed tokens are discarded and **not recorded** (outcome recorded as `latency=ttft` failure only). The recovery/graceful stream phases don't even apply the hedge (`ttft_g` is a flat 15 s at `:2744`, `:3125`).
- The fallback matrix exists but is *implicit and un-annotated*: explicit-request horizontals (`selector.py:391–404`) implement the "same model across providers" tier; intent-escalation tails implement sibling→higher tiers; quality floor/`min_quality_ratio` prunes mid-flight; graceful any-live is the emergency tier (budget-dead per R1, and **disabled for explicit requests** by default: `allow_graceful_fallback_on_explicit=False`). There is **no error-class-keyed matrix** (429→different provider first; context-overflow→truncate/retry; tools-400→capability-filtered tier) and **no provider-diversity constraint** — if the top-N ladder models all live on one provider, one outage consumes the entire chain in a single request.

---

## 4. What works (keep and protect)

| Area | Evidence |
|---|---|
| Error taxonomy & retryable classification | `_is_retryable_model_error` covers 401/403/405/408/429/5xx, 404-model, *and* provider-wrapped `400 "upstream request failed"` (`fallback.py:131–158`); context-overflow detection; non-retryable 400 hard-stop. |
| Soft-failure detection | `_analyze_success_body` — schema-aware empty-reply/tool-miss/HTML-page detection forces chain advance instead of serving garbage (`:199–281`), incl. malformed-JSON-as-transport (`ae63b43`). |
| Key layer | Multi-key weighted selection (headroom × latency × success-rate), RPM/RPD windows, 429 cooldown honoring `Retry-After`, auth-failure quarantine, in-flight caps, cancel-safe release on `BaseException` (`balancer.py`, `upstream.py`). |
| Health layer | Per-model + per-(model,key) EWMA, adaptive cooldown growth (5xx/504/429 differentiated), cooldown cleared on success, `health_reorder` keeps quality order while demoting non-responders (`catalog/health.py`). |
| Watchdogs | TTFT timeout (adaptive: `min(15s, max(3s, ewma*2+3))`), 300 s idle watchdog, per-intent deadlines & attempt budgets, `deadline_guard_seconds`, empty-stream → 502 (never 200+empty). |
| Streaming hygiene | Never bare `[DONE]`; `finish_reason=error` + error event framing; backpressure queue; `stream_failed` flag → guard failure + 499 log; `X-Accel-Buffering:no`. |
| Routing intelligence | Intent classification, ladder scoring, optimizer (intelligence × speed × health), LinUCB re-rank + exploration slot, sticky sessions, RL rewards keyed on status/TTFT, custom ladders, model pools, disabled-model filtering. |
| Observability | `TraceSpan` per attempt (classify/route/upstream/fallback_advance), request log ring, breaker + health snapshots in admin, self-heal loop (`heal_and_refresh`) with alerts. |
| Test discipline | 498 passing tests incl. resilience-regression and 504-cascade suites; deliberate invariants documented in comments (`F-05`…`F-18`, `NMK-*`). |

---

## 5. Critical deficiencies (prioritized)

| ID | Sev | Defect | Location | Impact |
|----|-----|--------|----------|--------|
| D1 | **P0** | Recovery phases budget-deadlocked against primary chain | `fallback.py:1187,1277,1664,1866,1897,1982,2611,3008,3020` | 503 despite healthy untried models — the production symptom |
| D2 | **P0** | Circuit-open *skip* recorded as transport failure (stream) | `fallback.py:2008–2009,2656,3038` vs `hub.py:210–214` | Breaker cascade/flapping; chain skews to few providers |
| D3 | **P0** | Two-tier violations; model-tier breaker dead; rate-limit→transport | `fallback.py:699,762,1383,1836,2055`; `circuit_breaker.py:46–53,104–151` | Healthy providers opened by model/429 events |
| D4 | **P0** | Global cooldown wipe + force-allow-all in recovery | `fallback.py:1665–1670,2612–2617` | Breaker defeated during incidents; cross-request interference |
| D5a | **P1** | Fresh phase rebuilds same chain ⇒ `fresh≈∅` | `fallback.py:1671–1689,2618–2636` | "Last-resort" is a no-op |
| D5b | **P1** | Unbounded `Retry-After` sleep in-loop | `backoff.py:24–25`; `fallback.py:1621–1651,2563–2575`; `upstream.py:162–175` | Deadline overrun, gate held, orphaned attempts |
| D5c | **P1** | `RateLimitedError` → JSONResponse unpack → 500 | `routes/openai.py:367–370` vs `:700,:707` | Wrong status + lost error body when user limits on |
| D5d | **P1** | Terminal 503 lacks `Retry-After`; `last_body` passed raw | `fallback.py:1904–1921,3336–3348` | Clients can't back off; provider text surfaces as gateway text |
| D6a | **P1** | No mid-stream upstream error-frame reconciliation | `fallback.py:2347–2359` | Corrupted streams recorded as successes |
| D6b | **P1** | No explicit `aclose()` propagation; cancel window in `upstream.stream` | `compat.py:361–378`, `openai.py:826+`, `fallback.py:2477–2479`, `upstream.py:363–372` | Key `in_flight`/socket leaks under disconnects |
| D6c | **P1** | `httpx.Timeout` read/pool/write = 300 s; passthrough unbounded | `upstream.py:60–73`, `openai.py:1085` | 5-minute stalls; no fail-fast on pool saturation |
| D7a | **P2** | "TTFT hedging" is sequential fail-fast, inconsistent on recovery paths | `fallback.py:2154–2167,2744,3125` | P99 = sum of two TTFTs; wasted billed tokens |
| D7b | **P2** | No error-class-keyed / provider-diverse fallback matrix | `selector.py`, `fallback.py:883–1199` | Correlated failure inside one chain |
| D7c | **P2** | Breaker/health/key state is per-process, no windowed rates | `circuit_breaker.py`, `health.py` | Replica skew; consecutive-count opens |
| D8 | **P2** | Generic `except Exception → 503 "Upstream request failed"` masks real bugs | `routes/openai.py:1140–1163` | Programming errors surface as retriable 503 |

---

## 6. Comparison with state of the art

| Axis | Potato (today) | LiteLLM | OpenRouter | RouteLLM | Envoy AI Filters | Verdict |
|---|---|---|---|---|---|---|
| **Two-tier circuit breaking** | Breaker + model cooldown exist; tier wiring broken (D2/D3); consecutive counts; per-process | Cooldown windows on model *and* provider groups; pre-call checks skip cooled targets | Implicit provider health + instant failover | n/a (score-based routing) | **Gold standard:** per-cluster circuit breakers + outlier detection (5xx/5xx-host success-rate over window) + active HC + `retry_budget` | Behind: adopt windowed success-rate + strict tier separation + probe-token semantics |
| **TTFT hedging** | Sequential fail-fast (`ttft_hedge_factor`), applied inconsistently | Streaming fallbacks before first token (sequential) | Sequential; retries only pre-first-token | Cascade scoring (pre-request) | **Route-level `hedge_policy`** (true parallel hedge with budget) | On par with OpenRouter; behind Envoy for P99 |
| **Adaptive fallback matrix** | Implicit: horizontals (explicit), escalation tails, quality floor, graceful any-live (dead: D1) | **Error-class-keyed `fallbacks` / `context_window_fallbacks` / `content_policy_fallbacks`** + same-model-across-providers | `provider.order`/`allow_fallbacks`: same model multi-provider first | Sibling/cheaper tier by score | `retry_priority` + outlier-aware cluster selection | Behind: needs error-class keying, provider diversity, live emergency tier |
| **SSE reconciliation & socket lifecycle** | `finish_reason=error` framing ✔ (OR parity); no upstream error-frame scan; GC-based cleanup; commit-on-first-chunk | Converts mid-stream exceptions to error events | Error event + `[DONE]` on provider death | n/a | Connection pooling/HC owns sockets | Framing: parity. Detection + deterministic cleanup + commit threshold: behind |

---

## 7. Architectural recommendations (target state)

### A. Explicit request plan with tiers (replaces ad-hoc `_chain` post-processing)

`_chain()` should return `list[Candidate(model, provider_id, tier, reason)]` where tiers are:

```
T0 sticky/pin head      (session affinity — keep, F-08)
T1 horizontal siblings  (same bare model, other providers — exists for explicit only)
T2 intent siblings      (same ladder, next-best)      ← enforce provider diversity:
                              no >2 consecutive same-provider; ≥2 providers when available
T3 intent escalation    (exists as tail)
T4 emergency frontier   (small, always-eligible strong set; bypasses quality floor,
                              NEVER bypasses allowed_models/free/disabled)
T5 structured degrade   (502/503 + Retry-After + machine-readable code)
```

Each phase gets its own **attempt reserve** (see remediation P0-1) and per-tier budgets; tier + provider are logged on `X-Potato-Tier`/`X-Potato-Fallback-Index` and on `TraceSpan`.

### B. Two-tier (three-signal) circuit breaking

```
Signal 1  Model cooldown   (per model, per provider-model)
          sources: 4xx (non-auth), 429, empty reply, model-404, quality soft-fails
          mechanism: ModelHealthStore cooldown (exists) + wire cb.model_fail (currently dead)
Signal 2  Provider transport breaker (per provider)
          sources: connect/read timeouts, 502/503/504, transport exceptions, mid-stream death
          mechanism: windowed success-rate: OPEN when failures≥N and rate≥R in W (e.g. 5/10 in 30s)
          states: CLOSED → OPEN(until T) → HALF_OPEN(single probe TOKEN, released on outcome)
          probe outcome is the ONLY transition trigger; skips/never-tried are neutral
Signal 3  Provider rate-limit budget (per provider, optional separate breaker)
          sources: 429 ratio, key-pool exhaustion, Retry-After-driven cooldowns
          mechanism: soft "capacity" state → prefer other providers, do NOT open transport circuit
```

Rules: `force_allow` gets a **min-interval** (e.g. 30 s/provider) so R4 cannot defeat it; probe failures keep exponential backoff but `_failures` decays on any success; `allow()` only called at *execution* time (once per actual attempt), `blocked()` for planning.

### C. Deadline & budget model

```
deadline      = min(client, request_deadline_seconds)
phase split   = 60% primary / 25% recovery / 15% reserve (structured error + logging)
attempt_budget = min(intent_budget, remaining − reserve_for_remaining_phases)
sleeps        = clamp(exp_backoff, Retry-After) ≤ remaining − guard; 429 → cool model, advance NOW (no sleep unless last candidate)
```

### D. TTFT speculative hedging (true)

- Race primary vs. secondary **only when**: P95 TTFT(primary) > hedge_threshold, or primary is on 2nd+ attempt, or `hedge_budget` allows (per-request max 1 hedge, global concurrency cap).
- Cancel loser at first content-bearing delta; record both TTFT samples; record abandoned usage if billed (`include_usage` when supported).
- Apply uniformly to chain **and** recovery phases; gate by `enable_ttft_hedging` + per-intent flag.
- Expected effect: P99 TTFB ≈ `min(T1, hedge_delay + T2)` instead of `T1 + T2`.

### E. Stream reconciliation & lifecycle

1. Scan `data:` payloads for top-level `"error"` (and `[DONE]`-without-usage anomalies) → `stream_failed`, model-tier cooldown, emit clean error frame.
2. **Commit threshold:** buffer up to first content delta (≤250 ms / ≤8 KB) before writing headers/bytes downstream; watchdog failover allowed pre-commit.
3. Deterministic cleanup: `try/finally: await source.aclose()` at every layer (`_gated_stream`, `normalize_sse_stream`, `robust_iter`→`rest`, producer→queue sentinel), and `resp.aclose()` in `upstream.stream`'s `except`.
4. `httpx.Timeout(read=upstream_timeout, connect=5, pool=10, write=30)`; passthrough gets the same deadline wrapper as routed traffic.

### F. Error contract (client-facing)

| Condition | Status | Body code | Headers |
|---|---|---|---|
| All candidates failed (upstream fault) | 502 | `all_providers_failed` | `Retry-After: 2..5`, last tier/provider |
| Capacity/gate/key-pool | 503 | `capacity_exhausted` | `Retry-After: 5..30` |
| Deadline | 504 | `request_deadline_exceeded` | — |
| Client-side per-user limit | 429 | `user_rate_limited` | `Retry-After` |

Never embed provider `last_body` verbatim at 5xx (structured `metadata.raw` capped instead) — that's what makes provider "Upstream Request Failed" text masquerade as gateway text.

---

Companion document: **`docs/ha-remediation-plan.md`** — priority plan with concrete diffs, tests, rollout, and the 99.99 % error-budget model.
