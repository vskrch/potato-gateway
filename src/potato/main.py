"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from potato import __version__
from potato.balancer import KeyPool
from potato.catalog import ModelRegistry
from potato.catalog.hub import ProviderHub
from potato.catalog.preferences import UserPreferences
from potato.catalog.providers import ProviderStore
from potato.config import Settings, get_settings
from potato.logging_setup import new_request_id, request_logs, setup_logging
from potato.routes import accounts, admin, analytics, claude, openai, public_chat, responses
from potato.routing import FallbackExecutor, IntentClassifier, ModelSelector, RoutingStats
from potato.safety import AccountGuard
from potato.upstream import UpstreamClient

logger = logging.getLogger("potato")


def _mount_vite_assets(app: FastAPI, dist_path: Path) -> bool:
    """Mount Vite assets only when the complete asset directory exists."""
    assets_path = dist_path / "assets"
    if not assets_path.is_dir():
        return False
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="vite-assets")
    return True


def _init_accounts(app: FastAPI, settings: Settings) -> None:
    """Attach AccountStore for signup / sessions / API keys.

    When ``settings.admin_password`` is configured, seed an active admin
    account on first boot so deploy.sh installs can sign in immediately.
    """
    app.state.accounts = None
    try:
        from potato.accounts.store import AccountStore
        from potato.catalog.db import get_db

        db = get_db(settings.sqlite_path)
        store = AccountStore(db)
        app.state.accounts = store
        logger.info("accounts store ready")

        # P0-1: first-time admin onboarding seed
        if settings.admin_password:
            _seed_admin(store, settings)
    except Exception:
        logger.exception("accounts store init failed")


def _seed_admin(store: Any, settings: Settings) -> None:
    """Idempotently seed an active admin account from env config.

    - User missing → create as admin, verify, activate, issue API key.
    - User exists, not admin → promote to admin.
    - User exists, not active → activate.
    - User exists, active, admin → no-op.
    """
    from potato.accounts.store import STATUS_ACTIVE, STATUS_UNVERIFIED

    email = settings.admin_email.strip().lower() or "admin@localhost"
    existing = store.get_user_by_email(email)
    if existing is None:
        try:
            user = store.create_user(
                email,
                settings.admin_password,
                role="admin",
                status=STATUS_UNVERIFIED,
            )
            # Verify → pending → active (matches the signup→verify→approve flow)
            store.mark_verified(user["id"])
            store.set_status(user["id"], STATUS_ACTIVE, approved_by="seed")
            key = store.issue_api_key(user["id"], name="seed-admin")
            logger.info(
                "seeded admin account email=%s key_prefix=%s — sign in at /dashboard",
                email,
                key.get("key_prefix"),
            )
        except Exception:
            logger.exception("failed to seed admin account email=%s", email)
        return
    # Existing user — promote / activate if needed
    if existing.get("role") != "admin":
        store.set_role(existing["id"], "admin")
        logger.info("promoted existing user email=%s to admin", email)
    if existing.get("status") != STATUS_ACTIVE:
        store.mark_verified(existing["id"])
        store.set_status(existing["id"], STATUS_ACTIVE, approved_by="seed")
        logger.info("activated existing admin email=%s", email)


def _init_analytics(app: FastAPI, settings: Settings) -> None:
    """Start analytics writer / retention / event bus when enabled.

    EventBus is always created so Live Feed can stream request-log events
    even when full analytics persistence is disabled.
    """
    from potato.analytics.events import EventBus

    app.state.event_bus = EventBus()
    app.state.trace_writer = None
    app.state.analytics_store = None
    app.state.retention_manager = None
    if not getattr(settings, "analytics_enabled", True):
        logger.info("analytics persistence disabled (live request feed still available)")
        return
    try:
        from potato.analytics.retention import RetentionManager
        from potato.analytics.store import AnalyticsStore
        from potato.analytics.writer import TraceWriter
        from potato.catalog.db import get_db

        db = get_db(settings.sqlite_path)
        bus = app.state.event_bus
        store = AnalyticsStore(db)

        on_flush = None
        hooks: list[Any] = []
        webhook_url = getattr(settings, "analytics_webhook_url", None)
        if webhook_url:
            from potato.analytics.webhook import WebhookBroadcaster

            hooks.append(WebhookBroadcaster(webhook_url).on_flush)
        otlp = getattr(settings, "analytics_otlp_endpoint", None)
        if otlp:
            from potato.analytics.otel import OTLPExporter

            hooks.append(OTLPExporter(otlp).on_flush)

        def _combined_flush(batch: Any) -> None:
            for h in hooks:
                try:
                    h(batch)
                except Exception:
                    logger.exception("analytics flush hook failed")

        if hooks:
            on_flush = _combined_flush

        writer = TraceWriter(
            db,
            batch_size=int(getattr(settings, "analytics_batch_size", 50) or 50),
            flush_interval=float(getattr(settings, "analytics_flush_interval", 1.0) or 1.0),
            event_bus=bus,
            on_flush=on_flush,
        )
        retention = RetentionManager(
            db,
            retention_days=int(getattr(settings, "analytics_retention_days", 7) or 7),
            rollup_retention_days=int(
                getattr(settings, "analytics_rollup_retention_days", 90) or 90
            ),
        )
        app.state.trace_writer = writer
        app.state.analytics_store = store
        app.state.retention_manager = retention
    except Exception:
        logger.exception("analytics init failed — continuing without analytics persistence")


def _configure_request_logs(app: FastAPI, settings: Settings) -> None:
    """Bind rotating request logs next to SQLite + live SSE publisher."""
    from potato.catalog.db import get_db
    from potato.logging_setup import default_log_dir, request_logs

    try:
        db = get_db(settings.sqlite_path)
    except Exception:
        db = None
        logger.exception("request log: sqlite unavailable for meta flag")

    def _on_add(entry: Any) -> None:
        bus = getattr(app.state, "event_bus", None)
        if bus is None:
            return
        try:
            bus.publish("request", entry.to_dict())
        except Exception:
            logger.debug("request log live publish failed", exc_info=True)

    request_logs.configure(
        max_entries=max(100, int(settings.request_log_size)),
        log_dir=default_log_dir(settings.sqlite_path),
        enabled=bool(getattr(settings, "request_file_logging", True)),
        max_file_bytes=int(getattr(settings, "request_log_max_bytes", 50 * 1024 * 1024)),
        retention_days=int(getattr(settings, "request_log_retention_days", 90)),
        db=db,
        on_add=_on_add,
    )
    st = request_logs.status()
    logger.info(
        "request file logging enabled=%s dir=%s active=%s max_file=%sB retention=%sd",
        st.get("enabled"),
        st.get("log_dir"),
        st.get("file_path"),
        st.get("max_file_bytes"),
        st.get("retention_days"),
    )


async def _start_analytics(app: FastAPI) -> None:
    writer = getattr(app.state, "trace_writer", None)
    retention = getattr(app.state, "retention_manager", None)
    if writer is not None:
        await writer.start()
    if retention is not None:
        await retention.start()


async def _stop_analytics(app: FastAPI) -> None:
    retention = getattr(app.state, "retention_manager", None)
    writer = getattr(app.state, "trace_writer", None)
    if retention is not None:
        await retention.stop()
    if writer is not None:
        await writer.stop()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    # max_entries finalized in _configure_request_logs during lifespan
    request_logs.configure(max_entries=max(100, int(settings.request_log_size)))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        store = ProviderStore.load(
            settings.providers_config_path,
            settings.providers_overlay_path,
            nim_base_url=settings.nim_base_url,
            nim_api_keys=list(settings.nim_api_keys),
            nim_rpm=settings.nim_rpm_limit,
            nim_rpd=settings.nim_rpd_limit,
            nim_max_in_flight=settings.nim_max_in_flight_per_key,
            sqlite_path=settings.sqlite_path,
            seed_free_presets=settings.sqlite_seed_free_presets,
        )
        hub = ProviderHub(store, settings)
        try:
            await hub.start()
        except Exception:
            logger.exception("provider hub startup failed — running degraded")

        # Check actual resolved keys (covers env alias, admin UI / SQLite, and NIM_API_KEYS)
        nim_cfg = hub.store.providers.get("nim")
        _nim_has_keys = bool(nim_cfg and nim_cfg.resolved_keys())
        _any_provider_has_keys = bool(hub.active_provider_ids())
        if not _nim_has_keys and not _any_provider_has_keys:
            logger.error(
                "No upstream API keys configured. Set NIM_API_KEYS in the environment "
                "or add a provider with keys via /admin/providers."
            )
        elif not _nim_has_keys:
            logger.info(
                "NIM_API_KEYS not set — routing through other configured provider(s): %s",
                sorted(hub.active_provider_ids()),
            )
        if not settings.proxy_api_keys and not settings.allow_insecure_auth:
            logger.warning(
                "PROXY_API_KEYS is empty and ALLOW_INSECURE_AUTH is false — "
                "all client requests will be rejected until you set proxy keys."
            )

        # Get default upstream/pool — create fail-closed if no providers have keys
        try:
            upstream = hub.default
            pool = hub.default_pool
        except RuntimeError:
            # No providers with keys — create fail-closed pool
            pool = KeyPool(
                api_keys=["placeholder-no-keys"],
                rpm_limit=settings.effective_rpm,
                rpd_limit=settings.nim_rpd_limit,
                max_in_flight_per_key=1,
                auth_fail_threshold=settings.auth_fail_threshold,
                auth_quarantine_seconds=settings.auth_quarantine_seconds,
            )
            upstream = UpstreamClient(
                base_url=settings.nim_base_url,
                pool=pool,
                timeout=settings.upstream_timeout,
                user_agent=settings.upstream_user_agent,
                proxy_url=settings.egress_proxy_url(),
                retry_backoff_base=settings.retry_backoff_base_seconds,
                retry_backoff_cap=settings.retry_backoff_cap_seconds,
            )
            await upstream.start()

        registry: ModelRegistry | None = None
        classifier = IntentClassifier(settings)
        selector: ModelSelector | None = None
        fallback: FallbackExecutor | None = None
        routing_stats = RoutingStats()
        refresh_task: asyncio.Task | None = None
        intel_task: asyncio.Task | None = None

        try:
            registry = ModelRegistry.from_settings(settings)
        except FileNotFoundError:
            logger.warning(
                "models catalog missing at %s — routing will be limited",
                settings.models_config_path,
            )

        guard = AccountGuard(settings, pool, capacity_hint=hub.gate_capacity() or None)
        hub._on_capacity_change = lambda: guard.resize_gate(hub.gate_capacity())

        # Load user preferences & custom model ladders (SQLite)
        from potato.catalog.db import get_db
        from potato.catalog.model_ladders import ModelLadderStore

        _db = get_db(settings.sqlite_path)
        preferences = UserPreferences(
            path=Path(".potato/user_preferences.json"),
            db_path=Path(settings.sqlite_path),
        )
        preferences.load()

        model_ladders = ModelLadderStore(_db)
        model_ladders.load()

        from potato.catalog.model_pools import ModelPoolStore

        model_pools = ModelPoolStore(_db)
        model_pools.load()

        # Initialize LinUCB RL Engine (NMK-RL-101)
        from potato.routing.rl_engine import LinUCBPolicyEngine, ModelLinUCBState

        rl_engine = LinUCBPolicyEngine()
        try:
            saved_rl = _db.load_rl_policy()
            with rl_engine._lock:
                for mid, sdict in saved_rl.items():
                    rl_engine._states[mid] = ModelLinUCBState.from_dict(sdict)
            if saved_rl:
                logger.info("loaded %d RL model policies from SQLite", len(saved_rl))
        except Exception:
            logger.exception("failed loading RL policies from SQLite")

        # Bind sticky ranking cache (SQLite) as early as possible
        try:
            if registry is not None:
                registry.rankings_sticky = True
                registry.bind_db(_db)
        except Exception:
            logger.exception("ranking cache bind failed — will recompute in-memory")

        if registry is not None:
            registry.ladder.provider_ids = set(hub.provider_ids)
            registry.rl_engine = rl_engine
            if hasattr(registry, "health") and registry.health is not None:
                registry.health._rl_engine = rl_engine
            selector = ModelSelector(
                registry,
                settings,
                preferences=preferences,
                model_ladders=model_ladders,
                rl_engine=rl_engine,
                model_pools=model_pools,
            )
            fallback = FallbackExecutor(
                upstream, registry, settings, stats=routing_stats, hub=hub, rl_engine=rl_engine
            )

            # NMK-G805: bind IntelFetcher + load YAML scoring weights
            from potato.catalog.intel_fetcher import IntelFetcher
            from potato.routing.optimizer import load_intent_weights

            intel_fetcher = IntelFetcher(
                cache_path=Path(getattr(settings, "intel_cache_path", ".potato/intel_cache.json")),
                ttl_hours=float(getattr(settings, "intel_fetch_ttl_hours", 6.0)),
                aa_api_key=(
                    getattr(settings, "artificial_analysis_api_key", "")
                    or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "")
                ),
            )
            registry.bind_intel_fetcher(intel_fetcher, settings)

            # Load intent optimizer weights from YAML scoring section
            if hasattr(registry, "_yaml_scoring_config"):
                load_intent_weights(registry._yaml_scoring_config.get("scoring", {}))

            # Defer initial catalog refresh to background — don't block startup
            async def _initial_refresh() -> None:
                try:
                    had_cache = bool(registry.ladder.frozen and registry.ladder._ladders)
                    # Skip docs on initial refresh — too slow for startup
                    ok = await registry.refresh_from_hub(
                        hub,
                        fetch_docs=False,
                        run_probes=False,
                        # Recompute rankings if no sticky cache; else keep frozen
                        recompute_rankings=not had_cache,
                    )
                    if not registry.ladder.frozen or not registry.dynamic_chains.get(
                        "coding_agentic"
                    ):
                        best = registry.recompute_rankings(persist=True)
                    else:
                        best = {
                            "coding_agentic": registry.dynamic_chains.get("coding_agentic", [])[:8],
                            "from_cache": True,
                        }
                    if not ok:
                        logger.warning(
                            "initial catalog refresh returned no models — "
                            "check provider API keys and base URLs"
                        )
                    logger.info(
                        "startup ready live=%s sticky_rankings=%s best_coding=%s",
                        len(registry.live_ids),
                        registry.ladder.frozen,
                        best.get("coding_agentic")
                        or best.get("best_coding")
                        or registry.dynamic_chains.get("coding_agentic", [])[:5],
                    )
                except Exception:
                    logger.exception("initial catalog refresh failed")

            asyncio.create_task(_initial_refresh())

            async def _refresh_loop() -> None:
                assert registry is not None
                from potato.resilience import heal_and_refresh

                cycle = 0
                every = max(1, int(settings.probe_every_n_refreshes))
                heal_every = max(30, int(getattr(settings, "self_heal_seconds", 120) or 120))
                last_heal = time.monotonic()
                last_learning_save = time.monotonic()
                learning_save_interval = 60.0
                while True:
                    # Wake at the earlier of catalog refresh vs self-heal interval
                    sleep_for = min(float(settings.catalog_refresh_seconds), float(heal_every))
                    await asyncio.sleep(max(15.0, sleep_for))
                    cycle += 1
                    now = time.monotonic()
                    # Lightweight heal often
                    if now - last_heal >= heal_every:
                        try:
                            empty = not registry.live_ids
                            report = await heal_and_refresh(
                                hub=hub,
                                registry=registry,
                                settings=settings,
                                force=empty,
                            )
                            last_heal = now
                            # NMK-304: auto-rerank when fallback rate is high
                            if routing_stats.should_rerank() and not empty:
                                try:
                                    registry.recompute_rankings(persist=True)
                                    logger.info(
                                        "adaptive rerank: %.0f%% recent fallback advances",
                                        sum(routing_stats._recent_advances)
                                        / len(routing_stats._recent_advances)
                                        * 100,
                                    )
                                except Exception:
                                    logger.exception("adaptive rerank failed")
                            if report.get("healed_models") or report.get("refreshed"):
                                logger.info("self-heal: %s", report)
                        except Exception:
                            logger.exception("self-heal loop failed")
                    # Full catalog refresh on its own cadence
                    if (
                        cycle % max(1, int(settings.catalog_refresh_seconds / max(15.0, sleep_for)))
                        == 0
                    ):
                        run_probes = settings.catalog_run_probes and cycle % every == 0
                        try:
                            # Background: update live model availability only —
                            # do NOT recompute sticky best-model rankings.
                            await registry.refresh_from_hub(
                                hub,
                                fetch_docs=settings.catalog_fetch_docs,
                                run_probes=run_probes,
                                recompute_rankings=False,
                            )
                        except Exception:
                            logger.exception("periodic catalog refresh failed")
                    # NMK-406: periodic learning persistence
                    now_ts = time.monotonic()
                    if now_ts - last_learning_save >= learning_save_interval:
                        try:
                            registry.learning.save()
                            last_learning_save = now_ts
                        except Exception:
                            logger.exception("periodic learning save failed")
                        # NMK-RL-101: persist LinUCB bandit policy to SQLite so
                        # the dashboard UI + routing weights survive restarts.
                        try:
                            with rl_engine._lock:
                                for mid, state in rl_engine._states.items():
                                    _db.upsert_rl_policy(
                                        mid,
                                        json.dumps(state.to_dict()),
                                        time.time(),
                                    )
                        except Exception:
                            logger.debug("RL policy persistence failed", exc_info=True)
                        try:
                            classifier.tiny_router.save_weights()
                        except Exception:
                            logger.debug("TinyRouter weights persistence failed", exc_info=True)

            refresh_task = asyncio.create_task(_refresh_loop())

            # NMK-G805: start intel refresh loop (periodic score recompute)
            intel_task = asyncio.create_task(registry.start_intel_refresh_loop())

        app.state.settings = settings
        app.state.hub = hub
        app.state.pool = pool
        app.state.upstream = upstream
        app.state.registry = registry
        app.state.classifier = classifier
        app.state.selector = selector
        app.state.fallback = fallback
        app.state.guard = guard
        app.state.routing_stats = routing_stats
        app.state.preferences = preferences
        app.state.model_ladders = model_ladders
        app.state.rl_engine = rl_engine
        app.state.model_pools = model_pools

        _init_accounts(app, settings)
        _init_analytics(app, settings)
        _configure_request_logs(app, settings)
        await _start_analytics(app)

        logger.info(
            "Potato v%s ready — providers=%s, routing=%s, analytics=%s, accounts=%s",
            __version__,
            [p.id for p in store.enabled_providers()],
            settings.routing_enabled,
            bool(getattr(app.state, "trace_writer", None)),
            bool(getattr(app.state, "accounts", None)),
        )
        try:
            yield
        finally:
            await _stop_analytics(app)
            if intel_task is not None:
                intel_task.cancel()
                with suppress(asyncio.CancelledError):
                    await intel_task
            if refresh_task is not None:
                refresh_task.cancel()
                with suppress(asyncio.CancelledError):
                    await refresh_task
            await hub.stop()

    app = FastAPI(
        title="Potato Gateway",
        description=(
            "Self-hosted OpenRouter-style gateway: NVIDIA NIM + any OpenAI-compatible "
            "providers with intelligent model routing."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    from potato.compat import openai_error

    @app.exception_handler(HTTPException)
    async def openai_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Unwrap FastAPI ``detail`` so clients see a top-level OpenAI ``error`` object."""
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            body: dict[str, Any] = detail
        elif isinstance(detail, dict):
            body = openai_error(
                str(detail.get("message") or detail)[:2000],
                code=str(detail.get("code") or "http_error"),
                type_=str(
                    detail.get("type")
                    or ("invalid_request_error" if exc.status_code < 500 else "server_error")
                ),
            )
        else:
            body = openai_error(
                str(detail)[:2000],
                code="http_error",
                type_=("invalid_request_error" if exc.status_code < 500 else "server_error"),
            )
        headers = dict(exc.headers or {})
        if exc.status_code in (401, 403):
            headers.setdefault("WWW-Authenticate", "Bearer")
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)

    @app.exception_handler(Exception)
    async def openai_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=openai_error(
                "Internal Server Error",
                code="internal_error",
                type_="server_error",
            ),
        )

    cors_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    # Credentials require explicit origins (never "*")
    cors_credentials = bool(cors_origins) and cors_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Potato-Model",
            "X-Potato-Intent",
            "X-Potato-Key-Id",
            "X-Potato-Route-Mode",
            "X-Potato-Fallback-Index",
            "X-Potato-Provider",
            "X-Potato-Context-Length",
            "X-Potato-Requested-Model",
            "X-Potato-Rule-Id",
            "X-Request-Id",
        ],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        rid = (
            request.headers.get("x-request-id")
            or request.headers.get("X-Request-Id")
            or new_request_id()
        )
        request.state.request_id = rid
        response = await call_next(request)
        response.headers.setdefault("X-Request-Id", rid)
        return response

    # Serve Vite build assets (dist/) with aggressive caching
    dist_path = Path(__file__).parent / "static" / "dist"
    _mount_vite_assets(app, dist_path)

    app.include_router(accounts.router)
    app.include_router(admin.router)
    app.include_router(openai.router)
    app.include_router(analytics.router)
    app.include_router(claude.router)
    app.include_router(responses.router)
    app.include_router(public_chat.router)

    @app.get("/manifest.json")
    @app.get("/sw.js")
    @app.get("/index.html")
    @app.get("/icon-192.png")
    @app.get("/icon-512.png")
    @app.get("/maskable-192.png")
    @app.get("/maskable-512.png")
    @app.get("/icon-192.svg")
    @app.get("/icon-512.svg")
    @app.get("/maskable-192.svg")
    @app.get("/maskable-512.svg")
    async def serve_static_root_file(request: Request) -> Any:
        """Serve root PWA manifest, service worker, and icon assets.

        ``/index.html`` is served here too so the service worker's
        ``cache.addAll`` install step succeeds (the shell must exist at the
        canonical PWA start URL). ``sw.js`` is never cached so browser
        service-worker update checks always see the latest bytes.
        """
        filename = request.url.path.lstrip("/")
        if filename == "sw.js":
            file_path = Path(__file__).parent / "static" / "dist" / "sw.js"
            if file_path.is_file():
                from fastapi.responses import FileResponse

                return FileResponse(
                    file_path,
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
                )
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if filename == "index.html":
            return _dashboard_html()
        file_path = Path(__file__).parent / "static" / "dist" / filename
        if file_path.is_file():
            from fastapi.responses import FileResponse

            return FileResponse(file_path)
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    def _dashboard_html() -> HTMLResponse:
        """Serve the compiled React TypeScript dashboard bundle."""
        dist_index = Path(__file__).parent / "static" / "dist" / "index.html"
        if dist_index.is_file():
            return HTMLResponse(
                content=dist_index.read_text(encoding="utf-8"),
                headers={"Cache-Control": "no-cache"},
            )
        return HTMLResponse(
            content="<h1>Potato Dashboard Assets Not Found</h1><p>Please run <code>cd frontend && npm run build</code> to generate the React bundle.</p>",
            status_code=404,
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/{path:path}", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        """Serve the web dashboard."""
        return _dashboard_html()

    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/chat/{path:path}", response_class=HTMLResponse)
    async def chat_ui() -> HTMLResponse:
        """Serve the standalone Claude-style chat app (same SPA, client-routed)."""
        return _dashboard_html()

    @app.get("/")
    async def root(request: Request) -> Any:
        """
        Browsers get the dashboard; API clients get the JSON discovery document.
        """
        accept = (request.headers.get("accept") or "").lower()
        # Prefer dashboard for human browsers
        if "text/html" in accept and "application/json" not in accept.split(",")[0]:
            return _dashboard_html()
        return {
            "name": "potato",
            "version": __version__,
            "dashboard": "/dashboard",
            "openai_base_url": "/v1",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "catalog": "/catalog",
            "providers": "/admin/providers",
            "status": "ok",
        }

    return app


# Module-level app for `uvicorn potato.main:app`
app = create_app()


def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if not settings.nim_api_keys and not settings.allow_insecure_auth:
        # Allow start if other providers may be configured in YAML/overlay
        logger.warning(
            "NIM_API_KEYS empty — ensure at least one provider has keys (NIM or /admin/providers)."
        )
    if not settings.proxy_api_keys and not settings.allow_insecure_auth:
        logger.warning(
            "PROXY_API_KEYS is empty — auto-enabling local development mode (ALLOW_INSECURE_AUTH=true). "
            "Set PROXY_API_KEYS in .env or configure keys at /dashboard to enforce authentication."
        )
        settings.allow_insecure_auth = True
    uvicorn.run(
        "potato.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    run()
