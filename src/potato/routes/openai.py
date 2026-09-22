"""OpenAI-compatible endpoints proxied to NVIDIA NIM."""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from potato.auth import require_active_user
from potato.catalog import ModelRegistry
from potato.compat import (
    inject_system_prompt,
    normalize_completion_json,
    normalize_sse_stream,
    openai_error,
    sanitize_chat_body,
    wrap_upstream_error,
)
from potato.config import Settings, get_settings
from potato.logging_setup import RequestLog, log_request_line, request_logs
from potato.routing import (
    FallbackExecutor,
    IntentClassifier,
    ModelSelector,
    RouteDecision,
)
from potato.routing.auto_router import (
    parse_auto_router_options,
    strip_router_client_fields,
)
from potato.safety import AccountGuard, RateLimitedError
from potato.upstream import UpstreamClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


def _enqueue_trace(request: Request, trace: Any) -> None:
    writer = getattr(request.app.state, "trace_writer", None)
    if writer is None:
        return
    try:
        writer.enqueue(trace)
    except Exception:
        logger.debug("trace enqueue failed", exc_info=True)


def _build_trace_base(
    request: Request,
    *,
    req_id: str,
    entry: RequestLog,
    body: dict[str, Any],
    proxy_token: str | None,
) -> Any:
    from potato.analytics.context import extract_request_context
    from potato.analytics.models import TraceRecord

    ctx_stats = extract_request_context(body)
    return TraceRecord(
        trace_id=req_id,
        created_at=entry.ts,
        method=entry.method,
        path=entry.path,
        client_ip=entry.client,
        api_key=proxy_token,
        user_id=getattr(getattr(request.state, "auth", None), "user_id", None),
        user_agent=entry.user_agent,
        model_requested=str(body.get("model") or "") or None,
        is_stream=bool(body.get("stream")),
        **ctx_stats,
    )


def _apply_timing(trace: Any, timing: dict[str, Any] | None) -> None:
    if not timing:
        return
    if timing.get("classify_ms") is not None:
        trace.classify_ms = timing["classify_ms"]
    if timing.get("route_ms") is not None:
        trace.route_ms = timing["route_ms"]
    if timing.get("intent_confidence") is not None:
        trace.intent_confidence = float(timing["intent_confidence"] or 0)
    if timing.get("intent_rule_id"):
        trace.intent_rule_id = timing["intent_rule_id"]


def _finalize_trace(
    request: Request,
    trace: Any,
    *,
    t0: float,
    status: int,
    decision: RouteDecision | None = None,
    model_routed: str | None = None,
    provider: str | None = None,
    fallback_index: int = 0,
    error: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    upstream_ttft_ms: float | None = None,
    upstream_total_ms: float | None = None,
    spans: list[Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> None:
    from potato.analytics.context import end_span_collection
    from potato.analytics.cost import estimate_cost

    settings = _settings(request)
    if not getattr(settings, "analytics_enabled", True):
        end_span_collection()
        return
    if trace is None:
        end_span_collection()
        return

    store = getattr(request.app.state, "analytics_store", None)
    overrides = store.cost_overrides_map() if store else None

    if spans is None:
        spans = end_span_collection()
    else:
        end_span_collection()

    _apply_timing(trace, timing)
    trace.duration_ms = (time.perf_counter() - t0) * 1000
    trace.status_code = status
    trace.success = 200 <= status < 400 and not error
    trace.error_message = error
    if decision is not None:
        trace.intent = decision.intent.value
        if not trace.intent_rule_id:
            trace.intent_rule_id = decision.rule_id
        trace.route_mode = decision.mode
        trace.model_requested = str(decision.requested_model or trace.model_requested or "") or None
        trace.chain = list(decision.chain or [])
        trace.chain_length = len(trace.chain) or 1
    if model_routed is not None:
        trace.model_routed = model_routed
    if provider is not None:
        trace.provider_id = provider
    trace.fallback_index = fallback_index
    upstream_prompt_tokens = int(prompt_tokens or 0)
    upstream_completion_tokens = int(completion_tokens or 0)
    trace.prompt_tokens = upstream_prompt_tokens
    trace.completion_tokens = upstream_completion_tokens
    trace.cached_tokens = int(cached_tokens or 0)
    if not trace.prompt_tokens and not trace.completion_tokens and trace.char_length:
        # Rough estimate when upstream omits usage — used for display only,
        # NOT for cost (cost must be on real upstream-reported tokens).
        trace.prompt_tokens = max(1, trace.char_length // 4)
    trace.total_tokens = trace.prompt_tokens + trace.completion_tokens
    trace.upstream_ttft_ms = upstream_ttft_ms
    trace.upstream_total_ms = upstream_total_ms
    if model_routed:
        # ponytail: cost is computed ONLY on real upstream-reported tokens.
        # When the upstream omits usage, cost is 0 (not a char-based estimate)
        # so the dashboard never shows inflated gateway-level cost.
        trace.estimated_cost_usd = estimate_cost(
            model_routed,
            upstream_prompt_tokens,
            upstream_completion_tokens,
            overrides=overrides,
        )
    if spans:
        trace.spans = list(spans)
    # Backfill classify/route ms from spans if not set
    for sp in trace.spans:
        if sp.span_type == "classify" and trace.classify_ms is None:
            trace.classify_ms = sp.duration_ms
        elif sp.span_type == "route" and trace.route_ms is None:
            trace.route_ms = sp.duration_ms
    _enqueue_trace(request, trace)


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def _upstream(request: Request) -> UpstreamClient:
    return request.app.state.upstream


def _routing_disabled(request: Request, settings: Settings) -> bool:
    if not settings.routing_enabled:
        return True
    flag = request.headers.get("x-potato-disable-route") or request.headers.get(
        "X-Potato-Disable-Route"
    )
    return flag in {"1", "true", "yes"}


async def _run_pre_router_interceptors(
    request: Request,
    body: dict[str, Any],
    *,
    intent: Any,
    registry: Any,
) -> dict[str, Any]:
    """NMK-EXT-200: Run the pre-router interceptor chain before core routing.

    Interceptors may mutate body["model"] from "auto" to a specific model.
    On any failure, body is returned unchanged (core router handles it).
    """
    model = str(body.get("model") or "").strip().lower()
    # Only run for auto-routed requests
    if model and model not in ("auto", ""):
        return body
    try:
        from potato.routing.interceptors import (
            CustomCatalogInterceptor,
            PromptUnderstandingInterceptor,
            run_interceptor_chain,
        )

        db = getattr(request.app.state, "_ext_db", None)
        if db is None:
            # Lazily get the DB for extensibility features
            try:
                from potato.catalog.db import get_db

                settings = _settings(request)
                db = get_db(settings.sqlite_path)
                request.app.state._ext_db = db
            except Exception:
                db = None

        interceptors: list = []
        # Interceptor 1: Custom Catalog Override
        if db is not None:
            features: dict[str, Any] = {}
            try:
                features = db.get_extensibility_features()
            except Exception:
                features = {}
            if features.get("custom_catalog_enabled"):
                interceptors.append(CustomCatalogInterceptor(db=db))
            # Interceptor 2: Prompt-Understanding Router
            if features.get("prompt_understanding_enabled"):
                upstream = _upstream(request)
                understanding_model = features.get("prompt_understanding_model", "")
                interceptors.append(
                    PromptUnderstandingInterceptor(
                        db=db,
                        upstream=upstream,
                        understanding_model=understanding_model,
                    )
                )
        if not interceptors:
            return body
        return await run_interceptor_chain(
            body, intent=intent, registry=registry, interceptors=interceptors
        )
    except Exception as exc:
        logger.debug("pre-router interceptor chain failed: %s — core router handles", exc)
        return body


async def _prepare_routed(
    request: Request,
    body: dict[str, Any],
    *,
    path: str,
    auto_opts: Any | None = None,
) -> tuple[
    dict[str, Any],
    RouteDecision | None,
    Any,
    str | None,
    dict[str, Any],
]:
    """
    Classify + select model chain. Returns (body, decision, guard_ctx, proxy_token, timing).
    When routing is off, decision is None and body may only get default_model.

    ``auto_opts`` should be parsed from the raw body before ``sanitize_chat_body``.
    """
    from potato.analytics.context import begin_span_collection
    from potato.analytics.models import TraceSpan

    begin_span_collection()
    timing: dict[str, Any] = {
        "classify_ms": None,
        "route_ms": None,
        "intent_confidence": 0.0,
        "intent_rule_id": None,
    }
    settings = _settings(request)
    proxy_token = require_active_user(request, settings).token or ""
    guard: AccountGuard = request.app.state.guard
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)

    if _routing_disabled(request, settings):
        mid = str(body.get("model") or settings.default_model or "").strip()
        if mid and registry is not None:
            disabled_hit = registry.resolve_live_id(mid, include_disabled=True)
            if disabled_hit in getattr(registry, "disabled_models", set()):
                raise ValueError("model_disabled")
        if not body.get("model") and settings.default_model:
            body = {**body, "model": settings.default_model}
        ctx = await guard.before_request(
            headers=request.headers, proxy_token=proxy_token, body=body
        )
        return body, None, ctx, proxy_token, timing

    classifier: IntentClassifier = request.app.state.classifier
    selector: ModelSelector | None = request.app.state.selector

    if selector is None or registry is None:
        if not body.get("model") and settings.default_model:
            body = {**body, "model": settings.default_model}
        ctx = await guard.before_request(
            headers=request.headers, proxy_token=proxy_token, body=body
        )
        return body, None, ctx, proxy_token, timing

    if not body.get("model"):
        body = {**body, "model": settings.default_model or "potato/auto"}

    # Prefer caller-supplied opts (parsed from raw body); fall back to body parse
    if auto_opts is None:
        auto_opts = parse_auto_router_options(body)

    t_classify = time.perf_counter()
    intent = classifier.classify(path=path, body=body, headers=request.headers)
    if settings.classify_mode == "rules_then_llm":
        upstream = _upstream(request)
        pool = request.app.state.pool
        pressure = pool.available_count() <= max(1, len(pool) // 4)
        chain = registry.chain_for_intent("chat_fast")
        fast = chain[0] if chain else None
        intent = await classifier.classify_maybe_llm(
            path=path,
            body=body,
            headers=request.headers,
            upstream=upstream,
            fast_model=fast,
            pool_pressure_high=pressure,
        )
    classify_ms = (time.perf_counter() - t_classify) * 1000
    timing["classify_ms"] = classify_ms
    timing["intent_confidence"] = float(intent.confidence)
    timing["intent_rule_id"] = intent.rule_id
    from potato.analytics.context import collect_span

    collect_span(
        TraceSpan(
            span_type="classify",
            started_at=t_classify,
            ended_at=t_classify + classify_ms / 1000.0,
            duration_ms=classify_ms,
            success=True,
            metadata={
                "intent": intent.intent.value,
                "confidence": intent.confidence,
                "rule_id": intent.rule_id,
            },
        )
    )

    try:
        ctx = await guard.before_request(headers=request.headers, proxy_token=proxy_token, body=body)
    except RateLimitedError:
        # D5c: re-raise so the caller maps this to 429. Returning a
        # JSONResponse here made the caller unpack a 5-tuple from a Response
        # → TypeError → 500.
        raise
    # Auto-router session model pin (OpenRouter sticky model selection)
    preferred_model = getattr(ctx, "preferred_model", None)

    # ── Pre-Router Interceptor Chain (NMK-EXT-200) ──────────────
    # Runs before core routing; may mutate body["model"] from "auto" to a
    # specific model. Core router stays completely unchanged.
    body = await _run_pre_router_interceptors(request, body, intent=intent, registry=registry)

    t_route = time.perf_counter()
    try:
        decision = selector.resolve(
            body.get("model"),
            intent,
            auto_opts=auto_opts,
            preferred_model=preferred_model,
        )
        # Capture LinUCB feature vector for this request — used by rank_chain_with_rl
        # to re-rank the chain and by record_feedback to update the bandit online.
        try:
            from potato.routing.rl_features import extract_feature_vector

            decision.feature_vector = extract_feature_vector(
                body, headers=request.headers, intent_name=intent.intent.value
            )
            if decision.feature_vector and selector and getattr(selector, "rl_engine", None):
                decision.chain = selector.rank_chain_with_rl(decision.chain, decision.feature_vector)
        except Exception:
            decision.feature_vector = None
        # Rough token estimate for context-length filtering (T13)
        # For multi-turn sessions, use cumulative context from previous turns
        try:
            msgs = body.get("messages") or body.get("input") or body.get("prompt") or ""
            char_n = sum(len(str(m)) for m in msgs) if isinstance(msgs, list) else len(str(msgs))
            max_tok = int(body.get("max_tokens") or body.get("max_completion_tokens") or 0)
            base_estimate = int(char_n / 3.5) + max_tok + 512
            # If we have session context, use the actual token count from previous turns
            session_ctx = guard.sticky.get_session_context(ctx.session_id)
            if session_ctx and session_ctx.turn_count > 0:
                # Use actual tokens from previous turns + current request estimate
                decision.estimated_tokens = session_ctx.total_prompt_tokens + base_estimate
            else:
                decision.estimated_tokens = base_estimate
        except Exception:
            decision.estimated_tokens = None
    except BaseException:
        # Gate was acquired above — release it before propagating the error
        await guard.after_request(ctx, success=False)
        raise
    route_ms = (time.perf_counter() - t_route) * 1000
    timing["route_ms"] = route_ms
    collect_span(
        TraceSpan(
            span_type="route",
            started_at=t_route,
            ended_at=t_route + route_ms / 1000.0,
            duration_ms=route_ms,
            success=True,
            metadata={
                "mode": decision.mode,
                "chain_len": len(decision.chain),
                "chain_head": decision.chain[0] if decision.chain else None,
                "estimated_tokens": decision.estimated_tokens,
            },
        )
    )
    # Drop client-only OpenRouter/Kilo fields before upstream
    body = strip_router_client_fields(body)
    return body, decision, ctx, proxy_token, timing


def _merge_headers(
    upstream_headers: dict[str, str],
    extra: dict[str, str],
) -> dict[str, str]:
    out = {k: v for k, v in upstream_headers.items() if k.lower() != "content-type"}
    out.update(extra)
    return out


@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    settings = _settings(request)
    require_active_user(request, settings)
    hub = getattr(request.app.state, "hub", None)
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)

    # Prefer unified live catalog from hub refresh (namespaced ids).
    # Admin-disabled models are omitted from the client-visible pool.
    if registry is not None and registry.live_ids:
        data = []
        for mid in sorted(registry.active_live_ids()):
            item: dict[str, Any] = {
                "id": mid,
                "object": "model",
                "created": 0,
                "owned_by": mid.split("/", 1)[0] if "/" in mid else "unknown",
            }
            data.append(registry.enrich_model_entry(item))
        if settings.inject_auto_model:
            autos = registry.synthetic_auto_models()
            data = [*autos, *data]
        return JSONResponse(content={"object": "list", "data": data})

    # Fallback: single default upstream
    upstream = _upstream(request)
    status, body, headers, _key = await upstream.request_json("GET", "/models")
    if isinstance(body, dict) and isinstance(body.get("data"), list) and registry is not None:
        if hub is not None:
            pid = "nim"
            namespaced = []
            for item in body["data"]:
                if isinstance(item, dict) and item.get("id"):
                    ns = hub.namespace(pid, str(item["id"]))
                    namespaced.append(registry.enrich_model_entry({**item, "id": ns}))
                else:
                    namespaced.append(item)
            body = {**body, "data": namespaced}
        else:
            registry._ingest_context_from_api_items(body["data"])
            body = {
                **body,
                "data": [
                    registry.enrich_model_entry(item) if isinstance(item, dict) else item
                    for item in body["data"]
                ],
            }
        if settings.inject_auto_model:
            auto = registry.synthetic_auto_model()
            ids = {item.get("id") for item in body["data"] if isinstance(item, dict)}
            if auto["id"] not in ids:
                body = {**body, "data": [auto, *body["data"]]}
    return JSONResponse(content=body, status_code=status, headers=headers)


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> JSONResponse:
    settings = _settings(request)
    require_active_user(request, settings)
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    hub = getattr(request.app.state, "hub", None)
    if registry is not None:
        from potato.routing.auto_router import is_auto_router_id

        # Future-proof: accept ANY claude-* model name (including ones not yet
        # in the alias list) so Claude Code CLI's pre-flight /v1/models check
        # always passes regardless of future Anthropic model releases.
        _is_claude = model_id.startswith("claude-")

        if is_auto_router_id(model_id) or model_id in {
            "auto",
            "potato/auto",
            "potato/auto-fast",
            "potato/auto-cheap",
            "potato/auto-coding",
            "potato/best",
            "openrouter/auto",
            "kilo/auto",
            "kilo-auto/frontier",
            "kilo-auto/balanced",
            "kilo-auto/efficient",
            "kilo-auto/free",
        } or registry.is_alias(model_id) or _is_claude:
            for m in registry.synthetic_auto_models():
                if m["id"] == model_id or (
                    model_id == "auto" and m["id"] in {"auto", "potato/auto"}
                ):
                    return JSONResponse(content=m)
            # Unknown claude-* model not in aliases — synthesize a valid entry
            # so the CLI's client-side validation passes.
            if _is_claude:
                return JSONResponse(content={
                    **registry.synthetic_auto_model(),
                    "id": model_id,
                    "owned_by": "anthropic",
                    "root": model_id,
                })
            return JSONResponse(content=registry.synthetic_auto_model())
    if registry is not None:
        resolved = registry.resolve_live_id(model_id) or model_id
        if resolved in registry.active_live_ids():
            item = {
                "id": resolved,
                "object": "model",
                "created": 0,
                "owned_by": resolved.split("/", 1)[0],
            }
            return JSONResponse(content=registry.enrich_model_entry(item))
        if resolved in registry.disabled_models:
            return JSONResponse(
                {
                    "error": {
                        "message": f"Model '{resolved}' is disabled in the pool.",
                        "type": "invalid_request_error",
                        "code": "model_disabled",
                    }
                },
                status_code=404,
            )
    if hub is not None:
        client, _pid, upstream_mid = hub.client_for_model(model_id)
        status, body, headers, _key = await client.request_json("GET", f"/models/{upstream_mid}")
        if registry is not None and isinstance(body, dict) and status < 400:
            body = registry.enrich_model_entry({**body, "id": model_id})
        return JSONResponse(content=body, status_code=status, headers=headers)
    upstream = _upstream(request)
    status, body, headers, _key = await upstream.request_json("GET", f"/models/{model_id}")
    if registry is not None and isinstance(body, dict) and status < 400:
        body = registry.enrich_model_entry(body)
    return JSONResponse(content=body, status_code=status, headers=headers)


def _client_ip(request: Request) -> str | None:
    xf = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xf:
        return xf.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _finish_log(
    entry: RequestLog,
    *,
    status: int,
    t0: float,
    model_routed: str | None = None,
    provider: str | None = None,
    intent: str | None = None,
    route_mode: str | None = None,
    fallback_index: int | None = None,
    stream: bool | None = None,
    error: str | None = None,
    model_requested: str | None = None,
    user_id: str | None = None,
) -> None:
    entry.status = status
    entry.duration_ms = (time.perf_counter() - t0) * 1000
    if model_routed is not None:
        entry.model_routed = model_routed
    if provider is not None:
        entry.provider = provider
    if intent is not None:
        entry.intent = intent
    if route_mode is not None:
        entry.route_mode = route_mode
    if fallback_index is not None:
        entry.fallback_index = fallback_index
    if stream is not None:
        entry.stream = stream
    if error is not None:
        entry.error = error
    if model_requested is not None:
        entry.model_requested = model_requested
    if user_id is not None:
        entry.user_id = user_id
    request_logs.add(entry)
    log_request_line(entry)


async def _chat_like(
    request: Request,
    *,
    upstream_path: str,
) -> JSONResponse | StreamingResponse:
    t0 = time.perf_counter()
    req_id = getattr(request.state, "request_id", None) or "noreq"
    entry = RequestLog(
        id=req_id,
        ts=time.time(),
        method=request.method,
        path=str(request.url.path),
        client=_client_ip(request),
        user_agent=(request.headers.get("user-agent") or "")[:160] or None,
    )
    upstream = _upstream(request)
    guard: AccountGuard = request.app.state.guard
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
    except Exception as exc:
        _finish_log(entry, status=400, t0=t0, error=f"invalid_json:{exc}")
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid JSON body: request body must be a JSON object",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
            status_code=400,
        )

    entry.model_requested = str(body.get("model") or "") or None
    entry.stream = bool(body.get("stream"))
    # Cursor sometimes sends huge payloads — log size only
    try:
        msg_n = len(body.get("messages") or body.get("input") or [])
        entry.notes.append(f"messages={msg_n}")
        if body.get("tools"):
            entry.notes.append(f"tools={len(body.get('tools') or [])}")
    except Exception:
        pass

    # OpenRouter plugins / session_id — parse from raw body before sanitize (F-02).
    auto_opts_raw = parse_auto_router_options(body)
    try:
        body = sanitize_chat_body(body)
    except ValueError as exc:
        if str(exc) == "n_not_supported":
            _finish_log(entry, status=400, t0=t0, error="n_not_supported")
            return JSONResponse(
                content=openai_error(
                    "Only n=1 is supported",
                    code="n_not_supported",
                    type_="invalid_request_error",
                    param="n",
                ),
                status_code=400,
            )
        raise

    # Universal system prompt — applied to all chat-style requests (anti-CJK-hallucination).
    sys_prompt = getattr(_settings(request), "default_system_prompt", "")
    body = inject_system_prompt(body, sys_prompt)

    timing: dict[str, Any] = {}
    proxy_token: str | None = None
    try:
        body, decision, ctx, proxy_token, timing = await _prepare_routed(
            request,
            body,
            path=upstream_path,
            auto_opts=auto_opts_raw,
        )
        auth = getattr(request.state, "auth", None)
        if auth is not None and getattr(auth, "user_id", None):
            entry.user_id = auth.user_id
    except RateLimitedError as exc:
        _finish_log(entry, status=429, t0=t0, error="user_rate_limited")
        return JSONResponse(content=exc.response, status_code=429)
    except Exception as exc:
        # Auth HTTPException propagates via FastAPI — re-raise
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            _finish_log(
                entry,
                status=exc.status_code,
                t0=t0,
                error=str(getattr(exc, "detail", "auth")),
            )
            raise
        # Concurrency gate exhaustion before stream/json phase (F-09)
        if isinstance(exc, RuntimeError) and "potato_pool_exhausted" in str(exc):
            _finish_log(entry, status=503, t0=t0, error=str(exc))
            return JSONResponse(
                content=guard.pool_exhausted_error(),
                status_code=503,
                headers={"Retry-After": "5"},
            )
        if isinstance(exc, ValueError) and str(exc) == "no_vision_model":
            _finish_log(entry, status=400, t0=t0, error="no_vision_model")
            return JSONResponse(
                content=openai_error(
                    "No vision-capable model is currently available.",
                    code="no_vision_model",
                    type_="invalid_request_error",
                ),
                status_code=400,
            )
        if isinstance(exc, ValueError) and str(exc) == "model_disabled":
            _finish_log(entry, status=400, t0=t0, error="model_disabled")
            return JSONResponse(
                content=openai_error(
                    "The requested model is disabled in the model pool.",
                    code="model_disabled",
                    type_="invalid_request_error",
                    param="model",
                ),
                status_code=400,
            )
        _finish_log(entry, status=500, t0=t0, error=str(exc))
        raise

    trace = None
    if getattr(_settings(request), "analytics_enabled", True):
        try:
            trace = _build_trace_base(
                request,
                req_id=req_id,
                entry=entry,
                body=body,
                proxy_token=proxy_token,
            )
            _apply_timing(trace, timing)
        except Exception:
            logger.debug("trace base build failed", exc_info=True)
            trace = None

    stream = bool(body.get("stream"))
    preferred = ctx.preferred_key_id
    chain_head = ""
    if decision is not None and decision.chain:
        chain_head = decision.chain[0]
        logger.info(
            "route req=%s path=%s requested=%s intent=%s mode=%s chain_head=%s chain_len=%s",
            req_id,
            upstream_path,
            decision.requested_model,
            decision.intent.value,
            decision.mode,
            chain_head,
            len(decision.chain),
        )

    try:
        if decision is not None:
            fallback: FallbackExecutor = request.app.state.fallback
            if stream:
                result = await fallback.execute_stream(
                    upstream_path,
                    body,
                    decision,
                    preferred_key_id=preferred,
                )
                try:
                    route_h = fallback.routing_headers(
                        decision,
                        model=result.model,
                        key_id=result.key.key_id if result.key else None,
                        fallback_index=result.fallback_index,
                        provider_id=getattr(result, "provider_id", None),
                    )
                    route_h["X-Request-Id"] = req_id
                    media = result.headers.get("content-type", "text/event-stream")
                    # Normalize SSE: canonicalize reasoning field to reasoning_content,
                    # phase-segregate thinking vs answer, fix model field
                    upstream_iter = normalize_sse_stream(
                        result.byte_iter, routed_model=result.model or None
                    )
                except Exception:
                    # Close upstream stream so key in_flight is released (F-07)
                    if hasattr(result.byte_iter, "aclose"):
                        with suppress(Exception):
                            await result.byte_iter.aclose()
                    raise
                key_id = result.key.key_id if result.key else None
                ok = 200 <= result.status_code < 300
                provider = route_h.get("X-Potato-Provider")

                pin_model = decision.mode in {
                    "auto",
                    "unknown_alias_as_auto",
                    "alias",
                }

                async def _gated_stream() -> Any:
                    err: str | None = None
                    try:
                        async for chunk in upstream_iter:
                            if await request.is_disconnected():
                                logger.info(
                                    "downstream client disconnected; aborting stream req=%s model=%s",
                                    req_id,
                                    result.model,
                                )
                                err = "client_disconnected"
                                break
                            yield chunk
                    except Exception as stream_exc:
                        err = str(stream_exc)
                        logger.warning(
                            "stream consumer error req=%s model=%s: %s",
                            req_id,
                            result.model,
                            stream_exc,
                        )
                        # Never bare [DONE] — emit finish_reason=error + error event
                        with suppress(Exception):
                            import json as _json

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
                                err[:500],
                                code="upstream_stream_error",
                                type_="server_error",
                            )
                            yield (b"data: " + _json.dumps(finish).encode("utf-8") + b"\n\n")
                            yield (b"data: " + _json.dumps(err_evt).encode("utf-8") + b"\n\n")
                            yield b"data: [DONE]\n\n"
                    finally:
                        # D6b: deterministic cleanup — close the normalized
                        # iterator (which closes robust_iter → upstream) before
                        # releasing the concurrency gate.
                        with suppress(Exception):
                            await upstream_iter.aclose()
                        # Check if robust_iter detected a mid-stream failure
                        if result.stream_failed and not err:
                            err = "mid_stream_failure"
                        await guard.after_request(
                            ctx,
                            key_id=key_id,
                            model_id=result.model,
                            success=ok and not err,
                            pin_model=pin_model,
                        )
                        # Update session context for multi-turn agentic tracking
                        usage = getattr(result, "usage", None) or {}
                        if ctx.session_id and ok and not err:
                            guard.sticky.update_session_context(
                                ctx.session_id,
                                prompt_tokens=int(
                                    usage.get("prompt_tokens") or result.prompt_tokens or 0
                                ),
                                completion_tokens=int(
                                    usage.get("completion_tokens") or result.completion_tokens or 0
                                ),
                                model_id=result.model,
                            )
                        status_final = result.status_code if not err else 499
                        _finish_log(
                            entry,
                            status=status_final,
                            t0=t0,
                            model_routed=result.model,
                            provider=provider,
                            intent=decision.intent.value,
                            route_mode=decision.mode,
                            fallback_index=result.fallback_index,
                            stream=True,
                            error=err,
                            model_requested=str(decision.requested_model or body.get("model") or "")
                            or None,
                        )
                        usage = getattr(result, "usage", None) or {}
                        _finalize_trace(
                            request,
                            trace,
                            t0=t0,
                            status=status_final,
                            decision=decision,
                            model_routed=result.model,
                            provider=provider,
                            fallback_index=result.fallback_index,
                            error=err,
                            prompt_tokens=int(
                                usage.get("prompt_tokens") or result.prompt_tokens or 0
                            ),
                            completion_tokens=int(
                                usage.get("completion_tokens") or result.completion_tokens or 0
                            ),
                            cached_tokens=int(
                                usage.get("cached_tokens") or result.cached_tokens or 0
                            ),
                            upstream_ttft_ms=result.upstream_ttft_ms,
                            timing=timing,
                        )

                return StreamingResponse(
                    _gated_stream(),
                    status_code=result.status_code,
                    media_type=media,
                    headers={
                        **_merge_headers(result.headers, route_h),
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            result_j = await fallback.execute_json(
                upstream_path,
                body,
                decision,
                preferred_key_id=preferred,
            )
            route_h = fallback.routing_headers(
                decision,
                model=result_j.model,
                key_id=result_j.key.key_id if result_j.key else None,
                fallback_index=result_j.fallback_index,
                provider_id=getattr(result_j, "provider_id", None),
            )
            route_h["X-Request-Id"] = req_id
            pin_model = decision.mode in {
                "auto",
                "unknown_alias_as_auto",
                "alias",
            }
            await guard.after_request(
                ctx,
                key_id=result_j.key.key_id if result_j.key else None,
                model_id=result_j.model,
                success=200 <= result_j.status_code < 300,
                pin_model=pin_model,
            )
            # Update session context for multi-turn agentic tracking
            if ctx.session_id and 200 <= result_j.status_code < 300:
                guard.sticky.update_session_context(
                    ctx.session_id,
                    prompt_tokens=result_j.prompt_tokens,
                    completion_tokens=result_j.completion_tokens,
                    model_id=result_j.model,
                )
            body_out = normalize_completion_json(result_j.body, routed_model=result_j.model or None)
            if result_j.status_code >= 400:
                body_out = wrap_upstream_error(body_out, status=result_j.status_code)
            err_msg = None if result_j.status_code < 400 else f"upstream_{result_j.status_code}"
            _finish_log(
                entry,
                status=result_j.status_code,
                t0=t0,
                model_routed=result_j.model,
                provider=route_h.get("X-Potato-Provider"),
                intent=decision.intent.value,
                route_mode=decision.mode,
                fallback_index=result_j.fallback_index,
                stream=False,
                model_requested=str(decision.requested_model or body.get("model") or "") or None,
                error=err_msg,
            )
            _finalize_trace(
                request,
                trace,
                t0=t0,
                status=result_j.status_code,
                decision=decision,
                model_routed=result_j.model,
                provider=route_h.get("X-Potato-Provider"),
                fallback_index=result_j.fallback_index,
                error=err_msg,
                prompt_tokens=result_j.prompt_tokens,
                completion_tokens=result_j.completion_tokens,
                cached_tokens=result_j.cached_tokens,
                upstream_total_ms=result_j.upstream_ms,
                timing=timing,
            )
            return JSONResponse(
                content=body_out,
                status_code=result_j.status_code,
                headers=_merge_headers(result_j.headers, route_h),
            )

        # Passthrough (routing disabled)
        if stream:
            # D6c: bound passthrough by the request deadline like routed traffic.
            import asyncio as _aio_pt

            _pt_deadline = float(getattr(_settings(request), "request_deadline_seconds", 300.0) or 300.0)
            status, byte_iter, headers, key = await _aio_pt.wait_for(
                upstream.stream(
                    "POST",
                    upstream_path,
                    json_body=body,
                    preferred_key_id=preferred,
                ),
                timeout=max(1.0, _pt_deadline),
            )
            media = headers.get("content-type", "text/event-stream")
            key_id = key.key_id
            ok = 200 <= status < 300

            async def _gated_passthrough() -> Any:
                err: str | None = None
                try:
                    async for chunk in byte_iter:
                        yield chunk
                except Exception as stream_exc:
                    err = str(stream_exc)
                    with suppress(Exception):
                        import json as _json

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
                            err[:500],
                            code="upstream_stream_error",
                            type_="server_error",
                        )
                        yield (b"data: " + _json.dumps(finish).encode("utf-8") + b"\n\n")
                        yield (b"data: " + _json.dumps(err_evt).encode("utf-8") + b"\n\n")
                        yield b"data: [DONE]\n\n"
                finally:
                    # D6b: close the upstream iterator on early exit.
                    with suppress(Exception):
                        await byte_iter.aclose()
                    await guard.after_request(ctx, key_id=key_id, success=ok and not err)
                    status_final = status if not err else 499
                    _finish_log(
                        entry,
                        status=status_final,
                        t0=t0,
                        model_routed=str(body.get("model") or ""),
                        stream=True,
                        route_mode="disabled",
                        error=err,
                    )
                    _finalize_trace(
                        request,
                        trace,
                        t0=t0,
                        status=status_final,
                        model_routed=str(body.get("model") or "") or None,
                        error=err,
                        timing=timing,
                    )

            return StreamingResponse(
                _gated_passthrough(),
                status_code=status,
                media_type=media,
                headers={
                    **{k: v for k, v in headers.items() if k.lower() != "content-type"},
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                    "X-Request-Id": req_id,
                },
            )

        # D6c: bound passthrough by the request deadline like routed traffic.
        import asyncio as _aio_ptj

        _ptj_deadline = float(getattr(_settings(request), "request_deadline_seconds", 300.0) or 300.0)
        status, resp_body, headers, key = await _aio_ptj.wait_for(
            upstream.request_json(
                "POST",
                upstream_path,
                json_body=body,
                preferred_key_id=preferred,
            ),
            timeout=max(1.0, _ptj_deadline),
        )
        await guard.after_request(ctx, key_id=key.key_id, success=200 <= status < 300)
        pt = ct = cached = 0
        if isinstance(resp_body, dict):
            usage = resp_body.get("usage")
            if isinstance(usage, dict):
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                cached = int(
                    usage.get("cached_tokens")
                    or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0
                )
        _finish_log(
            entry,
            status=status,
            t0=t0,
            model_routed=str(body.get("model") or ""),
            stream=False,
            route_mode="disabled",
        )
        _finalize_trace(
            request,
            trace,
            t0=t0,
            status=status,
            model_routed=str(body.get("model") or "") or None,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached_tokens=cached,
            timing=timing,
        )
        headers = {**headers, "X-Request-Id": req_id}
        if status >= 400:
            resp_body = wrap_upstream_error(resp_body, status=status)
        return JSONResponse(content=resp_body, status_code=status, headers=headers)
    except RuntimeError as exc:
        await guard.after_request(ctx, success=False)
        _finish_log(entry, status=503, t0=t0, error=str(exc), stream=stream)
        _finalize_trace(
            request,
            trace,
            t0=t0,
            status=503,
            decision=decision,
            error=str(exc),
            timing=timing,
        )
        if stream:
            from potato.compat import frame_sse_error

            async def _err_iter_rt():
                yield frame_sse_error(
                    str(exc) or "Upstream pool exhausted.",
                    code="potato_pool_exhausted",
                    status=503,
                )

            return StreamingResponse(
                _err_iter_rt(),
                status_code=200,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-Id": req_id,
                },
            )
        return JSONResponse(content=guard.pool_exhausted_error(), status_code=503)
    except Exception as exc:
        await guard.after_request(ctx, success=False)
        _finish_log(entry, status=503, t0=t0, error=str(exc), stream=stream)
        _finalize_trace(
            request,
            trace,
            t0=t0,
            status=503,
            decision=decision,
            error=str(exc),
            timing=timing,
        )
        logger.exception("chat path failed req=%s", req_id)
        if stream:
            from potato.compat import frame_sse_error

            async def _err_iter_ex():
                yield frame_sse_error(
                    "Upstream request failed — please retry.",
                    code="potato_internal_error",
                    status=503,
                    retry_after="5",
                )

            return StreamingResponse(
                _err_iter_ex(),
                status_code=200,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Retry-After": "5",
                    "X-Request-Id": req_id,
                },
            )
        return JSONResponse(
            content=openai_error(
                "Upstream request failed — please retry.",
                code="potato_internal_error",
                type_="server_error",
                metadata={"retry_after": 5},
            ),
            status_code=503,
            headers={"Retry-After": "5", "X-Request-Id": req_id},
        )
    except BaseException as exc:
        # CancelledError etc. — must release the concurrency gate
        with suppress(Exception):
            await guard.after_request(ctx, success=False)
        _finish_log(entry, status=499, t0=t0, error=type(exc).__name__, stream=stream)
        with suppress(Exception):
            _finalize_trace(
                request,
                trace,
                t0=t0,
                status=499,
                decision=decision,
                error=type(exc).__name__,
                timing=timing,
            )
        raise


@router.post("/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    return await _chat_like(request, upstream_path="/chat/completions")


@router.post("/completions", response_model=None)
async def completions(request: Request) -> JSONResponse | StreamingResponse:
    return await _chat_like(request, upstream_path="/completions")


@router.post("/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    t0 = time.perf_counter()
    req_id = getattr(request.state, "request_id", None) or "noreq"
    upstream = _upstream(request)
    guard: AccountGuard = request.app.state.guard
    settings = _settings(request)
    # Auth before body parse — reject unauthenticated clients early (T17)
    require_active_user(request, settings)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content=openai_error(
                "Invalid JSON body",
                code="invalid_json",
                type_="invalid_request_error",
            ),
            status_code=400,
        )
    try:
        body, decision, ctx, proxy_token, timing = await _prepare_routed(
            request, body, path="/embeddings"
        )
    except RateLimitedError as exc:
        return JSONResponse(content=exc.response, status_code=429)
    except RuntimeError as exc:
        if "potato_pool_exhausted" in str(exc):
            return JSONResponse(
                content=guard.pool_exhausted_error(),
                status_code=503,
                headers={"Retry-After": "5"},
            )
        raise
    except ValueError as exc:
        if str(exc) == "model_disabled":
            return JSONResponse(
                content=openai_error(
                    "The requested model is disabled in the model pool.",
                    code="model_disabled",
                    type_="invalid_request_error",
                    param="model",
                ),
                status_code=400,
            )
        if str(exc) == "no_vision_model":
            return JSONResponse(
                content=openai_error(
                    "No vision-capable model is currently available.",
                    code="no_vision_model",
                    type_="invalid_request_error",
                ),
                status_code=400,
            )
        raise
    preferred = ctx.preferred_key_id
    trace = None
    if getattr(_settings(request), "analytics_enabled", True):
        try:
            entry = RequestLog(
                id=req_id,
                ts=time.time(),
                method=request.method,
                path=str(request.url.path),
                client=_client_ip(request),
                user_agent=(request.headers.get("user-agent") or "")[:160] or None,
                model_requested=str(body.get("model") or "") or None,
            )
            trace = _build_trace_base(
                request,
                req_id=req_id,
                entry=entry,
                body=body,
                proxy_token=proxy_token,
            )
            _apply_timing(trace, timing)
        except Exception:
            logger.debug("embeddings trace build failed", exc_info=True)
            trace = None
    try:
        if decision is not None and request.app.state.fallback is not None:
            fallback: FallbackExecutor = request.app.state.fallback
            # If embeddings chain empty, passthrough requested model
            if not decision.chain and body.get("model"):
                status, resp_body, headers, key = await upstream.request_json(
                    "POST",
                    "/embeddings",
                    json_body=body,
                    preferred_key_id=preferred,
                )
                await guard.after_request(ctx, key_id=key.key_id, success=200 <= status < 300)
                _finalize_trace(
                    request,
                    trace,
                    t0=t0,
                    status=status,
                    decision=decision,
                    model_routed=str(body.get("model") or "") or None,
                    timing=timing,
                )
                return JSONResponse(content=resp_body, status_code=status, headers=headers)

            result = await fallback.execute_json(
                "/embeddings",
                body,
                decision,
                preferred_key_id=preferred,
            )
            route_h = fallback.routing_headers(
                decision,
                model=result.model,
                key_id=result.key.key_id if result.key else None,
                fallback_index=result.fallback_index,
            )
            await guard.after_request(
                ctx,
                key_id=result.key.key_id if result.key else None,
                success=200 <= result.status_code < 300,
            )
            _finalize_trace(
                request,
                trace,
                t0=t0,
                status=result.status_code,
                decision=decision,
                model_routed=result.model,
                provider=route_h.get("X-Potato-Provider"),
                fallback_index=result.fallback_index,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cached_tokens=result.cached_tokens,
                upstream_total_ms=result.upstream_ms,
                timing=timing,
            )
            return JSONResponse(
                content=result.body,
                status_code=result.status_code,
                headers=_merge_headers(result.headers, route_h),
            )

        status, resp_body, headers, key = await upstream.request_json(
            "POST",
            "/embeddings",
            json_body=body,
            preferred_key_id=preferred,
        )
        await guard.after_request(ctx, key_id=key.key_id, success=200 <= status < 300)
        _finalize_trace(
            request,
            trace,
            t0=t0,
            status=status,
            model_routed=str(body.get("model") or "") or None,
            timing=timing,
        )
        return JSONResponse(content=resp_body, status_code=status, headers=headers)
    except RuntimeError as exc:
        await guard.after_request(ctx, success=False)
        _finalize_trace(
            request, trace, t0=t0, status=503, decision=decision, error=str(exc), timing=timing
        )
        return JSONResponse(content=guard.pool_exhausted_error(), status_code=503)
    except Exception as exc:
        await guard.after_request(ctx, success=False)
        _finalize_trace(
            request, trace, t0=t0, status=500, decision=decision, error=str(exc), timing=timing
        )
        raise


# /v1/responses is owned by routes/responses.py (Responses→Chat translator).
