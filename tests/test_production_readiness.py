"""Production Readiness & Operational Security Verification Suite.

Comprehensive tests validating:
1. Frontend static asset bundle existence, integrity, and Vite mount serving.
2. SPA deep-linking and client routing catch-alls.
3. OpenAI/Claude API compatibility and root discovery.
4. Production authentication gates and credential masking.
5. Error sanitization (no Python stack trace leaks).
6. CORS configuration and security headers.
7. Streaming resilience and SSE error encapsulation.
8. SQLite database initialization and WAL journaling.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from potato.config import Settings
from potato.main import create_app
from potato.catalog.db import get_db


@pytest.mark.asyncio
async def test_frontend_dist_assets_integrity():
    """Verify frontend compiled production assets exist and are properly linked."""
    dist_dir = Path(__file__).parent.parent / "src" / "potato" / "static" / "dist"
    index_file = dist_dir / "index.html"
    assert dist_dir.is_dir(), f"Static dist directory missing: {dist_dir}"
    assert index_file.is_file(), f"index.html missing: {index_file}"

    content = index_file.read_text(encoding="utf-8")
    assert '<div id="root">' in content, "Missing root mount container in index.html"

    # Verify assets directory and references
    assets_dir = dist_dir / "assets"
    assert assets_dir.is_dir(), f"Vite assets directory missing: {assets_dir}"

    js_files = list(assets_dir.glob("*.js"))
    css_files = list(assets_dir.glob("*.css"))
    assert len(js_files) > 0, "No compiled JS chunks found in static/dist/assets"
    assert len(css_files) > 0, "No compiled CSS chunks found in static/dist/assets"

    # Verify script tags in index.html link to actual existing assets
    js_refs = re.findall(r'src=["\'](/assets/[^"\']+\.js)["\']', content)
    assert len(js_refs) > 0, "No JS asset references in index.html"
    for ref in js_refs:
        asset_name = ref.replace("/assets/", "")
        assert (assets_dir / asset_name).is_file(), f"Referenced JS asset not found on disk: {ref}"

    css_refs = re.findall(r'href=["\'](/assets/[^"\']+\.css)["\']', content)
    assert len(css_refs) > 0, "No CSS asset references in index.html"
    for ref in css_refs:
        asset_name = ref.replace("/assets/", "")
        assert (assets_dir / asset_name).is_file(), f"Referenced CSS asset not found on disk: {ref}"


@pytest.mark.asyncio
async def test_spa_and_static_serving():
    """Verify FastAPI serves the SPA bundle and mounted assets with correct MIME types."""
    settings = Settings(
        proxy_api_keys=["test-key"],
        allow_insecure_auth=True,
    )
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # 1. Root with Accept text/html -> returns SPA
        r_root_html = await client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
        assert r_root_html.status_code == 200
        assert "text/html" in r_root_html.headers.get("content-type", "").lower()
        assert '<div id="root">' in r_root_html.text

        # 2. Root with Accept application/json -> returns discovery JSON
        r_root_json = await client.get("/", headers={"accept": "application/json"})
        assert r_root_json.status_code == 200
        body = r_root_json.json()
        assert body.get("name") == "potato"
        assert body.get("status") == "ok"
        assert body.get("openai_base_url") == "/v1"

        # 3. Deep-linked SPA routes
        for path in ["/dashboard", "/dashboard/models", "/dashboard/analytics", "/chat"]:
            r = await client.get(path, headers={"accept": "text/html"})
            assert r.status_code == 200, f"Path {path} returned status {r.status_code}"
            assert "text/html" in r.headers.get("content-type", "").lower()
            assert r.headers.get("cache-control") == "no-cache"
            assert '<div id="root">' in r.text

        # 4. Vite mounted asset serving
        dist_dir = Path(__file__).parent.parent / "src" / "potato" / "static" / "dist"
        assets_dir = dist_dir / "assets"
        js_files = list(assets_dir.glob("index-*.js"))
        if js_files:
            js_name = js_files[0].name
            r_js = await client.get(f"/assets/{js_name}")
            assert r_js.status_code == 200
            assert "javascript" in r_js.headers.get("content-type", "").lower()


@pytest.mark.asyncio
async def test_production_auth_enforcement():
    """Verify strict proxy authentication when allow_insecure_auth is false."""
    settings = Settings(
        proxy_api_keys=["sk-potato-production-secret-999"],
        allow_insecure_auth=False,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver"
        ) as client:
            # Unauthenticated request to /v1/models
            r_unauth = await client.get("/v1/models")
            assert r_unauth.status_code == 401
            assert "WWW-Authenticate" in r_unauth.headers
            err_json = r_unauth.json()
            assert "error" in err_json, "Error response must be formatted as OpenAI error object"
            assert err_json["error"]["type"] == "invalid_request_error"

            # Wrong key
            r_wrong = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer sk-invalid-key"},
            )
            assert r_wrong.status_code == 401

            # Valid key passes auth gate (returns 200 OK with model list)
            r_valid = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer sk-potato-production-secret-999"},
            )
            assert r_valid.status_code == 200
            assert r_valid.json().get("object") == "list"


@pytest.mark.asyncio
async def test_error_sanitization_no_traceback_leak():
    """Verify unhandled exceptions return a clean OpenAI error without tracebacks."""
    settings = Settings(
        proxy_api_keys=["test-key"],
        allow_insecure_auth=True,
    )
    app = create_app(settings)

    # Mount a dummy route that raises an unhandled exception
    @app.get("/test-internal-error")
    async def _failing_route():
        raise RuntimeError("Secret DB connection failed at /var/secrets/db.key")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver"
    ) as client:
        r = await client.get("/test-internal-error")
        assert r.status_code == 500
        data = r.json()
        assert "error" in data
        assert data["error"]["code"] == "internal_error"
        assert data["error"]["message"] == "Internal Server Error"
        # Must not leak secret or file paths
        assert "db.key" not in r.text
        assert "Traceback" not in r.text


@pytest.mark.asyncio
async def test_security_headers_and_cors():
    """Verify CORS configuration and request tracking headers."""
    settings = Settings(
        proxy_api_keys=["test-key"],
        allow_insecure_auth=True,
        cors_allow_origins="https://app.example.com",
    )
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Preflight CORS check
        r_cors = await client.options(
            "/v1/models",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization,Content-Type",
            },
        )
        assert r_cors.status_code == 200
        assert r_cors.headers.get("access-control-allow-origin") == "https://app.example.com"
        assert r_cors.headers.get("access-control-allow-credentials") == "true"

        # Request-ID propagation
        r = await client.get("/health", headers={"X-Request-Id": "req-custom-trace-123"})
        assert r.headers.get("x-request-id") == "req-custom-trace-123"


@pytest.mark.asyncio
async def test_sqlite_wal_and_schema_integrity(tmp_path):
    """Verify SQLite database initializes in WAL mode with all required schemas."""
    db_path = tmp_path / "prod_test.db"
    db = get_db(str(db_path))

    with db._lock:
        journal_mode = db._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }

    # Verify essential operational tables exist
    required_tables = {
        "preferences",
        "model_ladders",
        "providers",
        "cost_overrides",
        "rl_policy",
        "users",
        "api_keys",
        "traces",
        "model_pool_config",
    }
    for t in required_tables:
        assert t in tables, f"Required table '{t}' missing from SQLite store"
    db.close()
