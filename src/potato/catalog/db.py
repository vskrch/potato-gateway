"""SQLite persistence for providers, preferences, and related gateway state.

Uses the stdlib ``sqlite3`` module (no extra deps). Default path:
``.potato/potato.db`` — set ``SQLITE_PATH`` / ``Settings.sqlite_path``.

Free-provider *templates* (base URLs) live in ``presets.py``; once you add
keys in the admin UI they are stored here so they survive restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    api_keys_json TEXT NOT NULL DEFAULT '[]',
    api_keys_env TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    rpm_limit REAL NOT NULL DEFAULT 40,
    rpd_limit INTEGER NOT NULL DEFAULT 2000,
    max_in_flight_per_key INTEGER NOT NULL DEFAULT 3,
    api_style TEXT NOT NULL DEFAULT 'openai',
    builtin INTEGER NOT NULL DEFAULT 0,
    model_whitelist_json TEXT NOT NULL DEFAULT '[]',
    model_blacklist_json TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS preferences (
    intent TEXT PRIMARY KEY,
    chain_json TEXT NOT NULL DEFAULT '[]',
    strict INTEGER NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ranking_cache (
    cache_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_ladders (
    model_id TEXT PRIMARY KEY,
    chain_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rl_policy (
    model_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_pool_config (
    model_id TEXT PRIMARY KEY,
    allowed_intents_json TEXT NOT NULL DEFAULT '[]',
    excluded_intents_json TEXT NOT NULL DEFAULT '[]',
    allow_auto_router INTEGER NOT NULL DEFAULT 1,
    note TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);
"""


class PotatoDB:
    """Thread-safe thin wrapper around a single SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we use explicit BEGIN
            timeout=15.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=15000")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate_provider_filter_columns()
            try:
                from potato.analytics.schema import migrate_analytics

                migrate_analytics(self._conn)
            except Exception:
                logger.exception("analytics schema migration failed")
                raise
            try:
                from potato.accounts.schema import migrate_accounts

                migrate_accounts(self._conn)
            except Exception:
                logger.exception("accounts schema migration failed")
                raise
        logger.info("sqlite ready at %s", self.path)

    def _migrate_provider_filter_columns(self) -> None:
        cols = {str(r[1]) for r in self._conn.execute("PRAGMA table_info(providers)").fetchall()}
        if "model_whitelist_json" not in cols:
            self._conn.execute(
                "ALTER TABLE providers ADD COLUMN model_whitelist_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "model_blacklist_json" not in cols:
            self._conn.execute(
                "ALTER TABLE providers ADD COLUMN model_blacklist_json TEXT NOT NULL DEFAULT '[]'"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── meta ────────────────────────────────────────────────────────

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ── providers ───────────────────────────────────────────────────

    def list_providers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM providers ORDER BY id").fetchall()
        return [self._provider_row(r) for r in rows]

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM providers WHERE id = ?", (provider_id.lower(),)
            ).fetchone()
        return self._provider_row(row) if row else None

    def upsert_provider(self, data: dict[str, Any]) -> None:
        pid = str(data["id"]).strip().lower()
        keys = data.get("api_keys") or []
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        wl = data.get("model_whitelist") or []
        bl = data.get("model_blacklist") or []
        payload = (
            pid,
            str(data.get("name") or pid),
            str(data.get("base_url") or "").rstrip("/"),
            json.dumps(list(keys)),
            data.get("api_keys_env"),
            1 if data.get("enabled", True) else 0,
            float(data.get("rpm_limit", 40)),
            int(data.get("rpd_limit", 2000)),
            int(data.get("max_in_flight_per_key", 3)),
            str(data.get("api_style") or "openai"),
            1 if data.get("builtin") else 0,
            json.dumps(list(wl)),
            json.dumps(list(bl)),
            float(data.get("updated_at") or time.time()),
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO providers (
                    id, name, base_url, api_keys_json, api_keys_env, enabled,
                    rpm_limit, rpd_limit, max_in_flight_per_key, api_style,
                    builtin, model_whitelist_json, model_blacklist_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    base_url = excluded.base_url,
                    api_keys_json = excluded.api_keys_json,
                    api_keys_env = excluded.api_keys_env,
                    enabled = excluded.enabled,
                    rpm_limit = excluded.rpm_limit,
                    rpd_limit = excluded.rpd_limit,
                    max_in_flight_per_key = excluded.max_in_flight_per_key,
                    api_style = excluded.api_style,
                    builtin = excluded.builtin,
                    model_whitelist_json = excluded.model_whitelist_json,
                    model_blacklist_json = excluded.model_blacklist_json,
                    updated_at = excluded.updated_at
                """,
                payload,
            )

    def delete_provider(self, provider_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM providers WHERE id = ?", (provider_id.lower(),))
            return cur.rowcount > 0

    def replace_all_providers(self, providers: list[dict[str, Any]]) -> None:
        """Atomic rewrite used after bulk load / migration."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("DELETE FROM providers")
                for data in providers:
                    pid = str(data["id"]).strip().lower()
                    keys = data.get("api_keys") or []
                    if isinstance(keys, str):
                        keys = [k.strip() for k in keys.split(",") if k.strip()]
                    wl = data.get("model_whitelist") or []
                    bl = data.get("model_blacklist") or []
                    self._conn.execute(
                        """
                        INSERT INTO providers (
                            id, name, base_url, api_keys_json, api_keys_env, enabled,
                            rpm_limit, rpd_limit, max_in_flight_per_key, api_style,
                            builtin, model_whitelist_json, model_blacklist_json,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pid,
                            str(data.get("name") or pid),
                            str(data.get("base_url") or "").rstrip("/"),
                            json.dumps(list(keys)),
                            data.get("api_keys_env"),
                            1 if data.get("enabled", True) else 0,
                            float(data.get("rpm_limit", 40)),
                            int(data.get("rpd_limit", 2000)),
                            int(data.get("max_in_flight_per_key", 3)),
                            str(data.get("api_style") or "openai"),
                            1 if data.get("builtin") else 0,
                            json.dumps(list(wl)),
                            json.dumps(list(bl)),
                            float(data.get("updated_at") or time.time()),
                        ),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _provider_row(row: sqlite3.Row) -> dict[str, Any]:
        def _json_list(col: str) -> list[str]:
            try:
                raw = row[col]
                data = json.loads(raw or "[]")
            except (json.JSONDecodeError, IndexError, KeyError):
                return []
            if not isinstance(data, list):
                return []
            return [str(x) for x in data if str(x).strip()]

        return {
            "id": row["id"],
            "name": row["name"],
            "base_url": row["base_url"],
            "api_keys": _json_list("api_keys_json"),
            "api_keys_env": row["api_keys_env"],
            "enabled": bool(row["enabled"]),
            "rpm_limit": float(row["rpm_limit"]),
            "rpd_limit": int(row["rpd_limit"]),
            "max_in_flight_per_key": int(row["max_in_flight_per_key"]),
            "api_style": row["api_style"] or "openai",
            "builtin": bool(row["builtin"]),
            "model_whitelist": _json_list("model_whitelist_json"),
            "model_blacklist": _json_list("model_blacklist_json"),
            "updated_at": float(row["updated_at"] or 0),
        }

    # ── preferences ─────────────────────────────────────────────────

    def list_preferences(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM preferences ORDER BY intent").fetchall()
        out = []
        for row in rows:
            try:
                chain = json.loads(row["chain_json"] or "[]")
            except json.JSONDecodeError:
                chain = []
            out.append(
                {
                    "intent": row["intent"],
                    "chain": list(chain) if isinstance(chain, list) else [],
                    "strict": bool(row["strict"]),
                    "note": row["note"] or "",
                    "updated_at": float(row["updated_at"] or 0),
                }
            )
        return out

    def upsert_preference(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO preferences (intent, chain_json, strict, note, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(intent) DO UPDATE SET
                    chain_json = excluded.chain_json,
                    strict = excluded.strict,
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (
                    str(data["intent"]),
                    json.dumps(list(data.get("chain") or [])),
                    1 if data.get("strict") else 0,
                    str(data.get("note") or ""),
                    float(data.get("updated_at") or time.time()),
                ),
            )

    def delete_preference(self, intent: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM preferences WHERE intent = ?", (intent,))
            return cur.rowcount > 0

    def clear_preferences(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM preferences")

    # ── ranking / best-models cache ─────────────────────────────────

    def get_ranking_cache(self, cache_key: str = "default") -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json, updated_at FROM ranking_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        data["_updated_at"] = float(row["updated_at"] or 0)
        return data

    def set_ranking_cache(self, payload: dict[str, Any], *, cache_key: str = "default") -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ranking_cache (cache_key, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (cache_key, json.dumps(payload), time.time()),
            )

    def clear_ranking_cache(self, cache_key: str | None = None) -> None:
        with self._lock:
            if cache_key:
                self._conn.execute("DELETE FROM ranking_cache WHERE cache_key = ?", (cache_key,))
            else:
                self._conn.execute("DELETE FROM ranking_cache")

    # ── extensibility features (NMK-EXT-501) ──────────────────────

    def get_extensibility_features(self) -> dict[str, Any]:
        """Read the extensibility feature toggles from meta."""
        raw = self.get_meta("extensibility_features")
        if not raw:
            return {
                "prompt_understanding_enabled": False,
                "prompt_understanding_model": "",
                "custom_catalog_enabled": False,
                "ollama_enabled": True,
                "opencode_go_enabled": True,
            }
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            return data
        except json.JSONDecodeError:
            return {}

    def set_extensibility_features(self, features: dict[str, Any]) -> None:
        """Persist extensibility feature toggles to meta."""
        self.set_meta("extensibility_features", json.dumps(features))

    # ── custom catalog mappings (NMK-EXT-301) ──────────────────────

    def get_custom_catalog_mappings(self) -> dict[str, str]:
        """Read intent → model_id mappings for the custom catalog override."""
        raw = self.get_meta("custom_catalog_mappings")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            return {str(k): str(v) for k, v in data.items() if v}
        except json.JSONDecodeError:
            return {}

    def set_custom_catalog_mappings(self, mappings: dict[str, str]) -> None:
        """Persist intent → model_id mappings to meta."""
        self.set_meta("custom_catalog_mappings", json.dumps(mappings))

    # ── model ladders (NMK-LAD-101) ─────────────────────────────

    def list_model_ladders(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT model_id, chain_json, note, updated_at FROM model_ladders ORDER BY model_id"
            )
            out: list[dict[str, Any]] = []
            for r in cur.fetchall():
                try:
                    chain = json.loads(str(r["chain_json"] or "[]"))
                    if not isinstance(chain, list):
                        chain = []
                except json.JSONDecodeError:
                    chain = []
                out.append(
                    {
                        "model_id": str(r["model_id"]),
                        "chain": chain,
                        "note": str(r["note"] or ""),
                        "updated_at": float(r["updated_at"] or 0.0),
                    }
                )
            return out

    def upsert_model_ladder(self, data: dict[str, Any]) -> None:
        model_id = str(data.get("model_id") or "").strip()
        if not model_id:
            return
        chain_json = json.dumps(list(data.get("chain") or []))
        note = str(data.get("note") or "")
        updated_at = float(data.get("updated_at") or time.time())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO model_ladders (model_id, chain_json, note, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    chain_json=excluded.chain_json,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (model_id, chain_json, note, updated_at),
            )

    def delete_model_ladder(self, model_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM model_ladders WHERE model_id = ?", (model_id,))

    def clear_model_ladders(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM model_ladders")

    # ── RL Policy Storage (NMK-RL-101) ──────────────────────────────────
    def load_rl_policy(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT model_id, payload_json FROM rl_policy").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            try:
                out[str(r["model_id"])] = json.loads(str(r["payload_json"] or "{}"))
            except Exception:
                continue
        return out

    def upsert_rl_policy(self, model_id: str, payload_json: str, updated_at: float) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO rl_policy (model_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (model_id, payload_json, updated_at),
            )

    def clear_rl_policy(self, model_id: str | None = None) -> None:
        with self._lock:
            if model_id:
                self._conn.execute("DELETE FROM rl_policy WHERE model_id = ?", (model_id,))
            else:
                self._conn.execute("DELETE FROM rl_policy")

    # ── Model Pool Gating Config Storage ──────────────────────────────
    def load_model_pool_configs(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_id, allowed_intents_json, excluded_intents_json, allow_auto_router, note, updated_at FROM model_pool_config"
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            mid = str(r["model_id"])
            try:
                allowed = json.loads(str(r["allowed_intents_json"] or "[]"))
            except Exception:
                allowed = []
            try:
                excluded = json.loads(str(r["excluded_intents_json"] or "[]"))
            except Exception:
                excluded = []
            out[mid] = {
                "model_id": mid,
                "allowed_intents": allowed if isinstance(allowed, list) else [],
                "excluded_intents": excluded if isinstance(excluded, list) else [],
                "allow_auto_router": bool(r["allow_auto_router"]),
                "note": str(r["note"] or ""),
                "updated_at": float(r["updated_at"] or 0.0),
            }
        return out

    def upsert_model_pool_config(
        self,
        model_id: str,
        allowed_intents_json: str,
        excluded_intents_json: str,
        allow_auto_router: int,
        note: str,
        updated_at: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO model_pool_config (
                    model_id, allowed_intents_json, excluded_intents_json, allow_auto_router, note, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    allowed_intents_json=excluded.allowed_intents_json,
                    excluded_intents_json=excluded.excluded_intents_json,
                    allow_auto_router=excluded.allow_auto_router,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (
                    model_id,
                    allowed_intents_json,
                    excluded_intents_json,
                    allow_auto_router,
                    note,
                    updated_at,
                ),
            )

    def delete_model_pool_config(self, model_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM model_pool_config WHERE model_id = ?", (model_id,))


# Process-wide cache so ProviderStore + UserPreferences share one connection.
_DB_CACHE: dict[str, PotatoDB] = {}
_DB_LOCK = threading.Lock()


def get_db(path: str | Path) -> PotatoDB:
    key = str(Path(path).resolve())
    with _DB_LOCK:
        db = _DB_CACHE.get(key)
        if db is None:
            db = PotatoDB(path)
            _DB_CACHE[key] = db
        return db
