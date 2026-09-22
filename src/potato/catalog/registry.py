"""Versioned model catalog: docs + live API + family preferences + probes."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from potato.catalog.aliases import normalize_model_name
from potato.catalog.context import (
    enrich_model_dict,
    extract_context_length,
    merge_context,
    parse_context_from_text,
)
from potato.catalog.docs_fetcher import DocModel, enrich_publishers, fetch_models_md
from potato.catalog.health import ModelHealthStore
from potato.catalog.ladder import LadderService
from potato.catalog.learning import LearningStore
from potato.catalog.prober import ProbeBudget, load_snapshot, probe_models, save_snapshot
from potato.catalog.schema import (
    AliasTarget,
    ModelsCatalog,
    catalog_from_dict,
    parse_alias_value,
)

if TYPE_CHECKING:
    from potato.upstream import UpstreamClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(
        self,
        catalog: ModelsCatalog,
        *,
        strict_catalog: bool = False,
        health: ModelHealthStore | None = None,
        snapshot_path: Path | None = None,
        docs_url: str = "https://build.nvidia.com/models.md",
        probe_budget_per_hour: int = 8,
        enrich_doc_details: bool = True,
    ) -> None:
        self.catalog = catalog
        self.strict_catalog = strict_catalog
        self.health = health or ModelHealthStore()
        self.learning = LearningStore(
            path=(snapshot_path or Path(".potato/catalog_snapshot.json")).parent / "learning.json"
        )
        self.learning.load()
        self.ladder = LadderService(health=self.health, learning=self.learning)
        self._apply_catalog_policy()
        self.live_ids: set[str] = set()
        # Admin model-pool customization: discovered models opted out of routing
        self.disabled_models: set[str] = set()
        # Lowercase namespaced id → original-case namespaced id (for upstream round-trip)
        self._original_ids: dict[str, str] = {}
        self.context_by_model: dict[str, int] = {}
        self.probed_ok: set[str] = set()
        self.doc_models: list[DocModel] = []
        self.dynamic_chains: dict[str, list[str]] = {}
        self.last_refresh_at: float | None = None
        self.last_refresh_ok: bool = False
        self.last_docs_ok: bool = False
        self._yaml_path: Path | None = None
        self.snapshot_path = snapshot_path or Path(".potato/catalog_snapshot.json")
        self.docs_url = docs_url
        self.probe_budget = ProbeBudget(probe_budget_per_hour)
        self.enrich_doc_details = enrich_doc_details
        # Sticky rankings: precompute once, serve until explicit refresh
        self.rankings_sticky: bool = True
        self.rankings_cache_key: str = "default"
        self._db: Any = None
        # Cache for intent_candidates (invalidated on live_ids/disabled_models change)
        self._intent_candidates_cache: dict[str, list[str]] = {}
        self._intent_candidates_key: tuple[frozenset[str], frozenset[str]] | None = None
        # Intel fetcher + score cache lifecycle (NMK-G804)
        self._intel_fetcher: Any = None
        self._intel_settings: Any = None
        self._yaml_scoring_config: dict = {}
        self._load_disk_snapshot()
        if self.live_ids:
            self.ladder.set_docs(self.doc_models)
            # Prefer frozen rebuild; cache load may override in bind_db()
            self.ladder.rebuild(self.active_live_ids(), freeze=True)
            self._sync_chains_from_ladder()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        strict_catalog: bool = False,
        snapshot_path: Path | None = None,
        docs_url: str = "https://build.nvidia.com/models.md",
        probe_budget_per_hour: int = 8,
    ) -> ModelRegistry:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"models catalog not found: {p}")
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        reg = cls(
            catalog_from_dict(data),
            strict_catalog=strict_catalog,
            snapshot_path=snapshot_path,
            docs_url=docs_url,
            probe_budget_per_hour=probe_budget_per_hour,
        )
        reg._yaml_path = p
        reg._yaml_scoring_config = reg._load_yaml_scoring_config()
        reg._sync_score_cache_once()
        if reg.live_ids:
            reg.recompute_rankings(persist=False)
        return reg

    def _apply_catalog_policy(self) -> None:
        """Push YAML family / primary_family prefs into LadderService."""
        fam = self.catalog.families
        primary: dict[str, str] = {
            "chat_fast": fam.chat_primary,
            "coding_agentic": fam.coding_primary,
            "reasoning": fam.chat_primary,
            "long_horizon": fam.coding_primary,
            "vision": fam.coding_primary,
            "embeddings": fam.chat_primary,
        }
        for intent, ic in self.catalog.intents.items():
            if ic.primary_family:
                primary[intent] = ic.primary_family
        self.ladder.apply_catalog_policy(
            primary_by_intent=primary,
            fallback_families=list(fam.fallbacks),
        )

    @classmethod
    def from_settings(cls, settings: Any) -> ModelRegistry:
        path = Path(settings.models_config_path)
        candidates = [
            path,
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
            # Packaged default shipped inside the wheel
            Path(__file__).resolve().parent / "data" / "models.yaml",
        ]
        try:
            from importlib import resources

            pkg = resources.files("potato") / "data" / "models.yaml"
            if pkg.is_file():
                candidates.insert(0, Path(str(pkg)))
        except Exception:
            pass
        resolved = None
        for c in candidates:
            try:
                if c.is_file():
                    resolved = c
                    break
            except Exception:
                continue
        if resolved is None:
            raise FileNotFoundError(
                f"models catalog not found (tried {settings.models_config_path} "
                "and packaged potato/data/models.yaml)"
            )
        snap = Path(getattr(settings, "catalog_snapshot_path", ".potato/catalog_snapshot.json"))
        return cls.from_yaml(
            resolved,
            strict_catalog=settings.strict_catalog,
            snapshot_path=snap,
            docs_url=getattr(settings, "catalog_docs_url", "https://build.nvidia.com/models.md"),
            probe_budget_per_hour=int(getattr(settings, "probe_budget_per_hour", 8)),
        )

    def _load_disk_snapshot(self) -> None:
        data = load_snapshot(self.snapshot_path)
        if not data:
            return
        self.live_ids = {str(x).strip().lower() for x in (data.get("live_ids") or [])}
        raw_ctx = data.get("context_by_model") or {}
        if isinstance(raw_ctx, dict):
            self.context_by_model = {
                str(k): int(v)
                for k, v in raw_ctx.items()
                if isinstance(v, (int, float)) and int(v) > 0
            }
        self.probed_ok = set(data.get("probed_ok") or [])
        self.dynamic_chains = {k: list(v) for k, v in (data.get("dynamic_chains") or {}).items()}
        logger.info(
            "loaded catalog snapshot (%s live ids, %s dynamic intents)",
            len(self.live_ids),
            len(self.dynamic_chains),
        )

    def _persist_snapshot(self) -> None:
        """Save catalog snapshot to disk. Non-blocking when called from async context."""
        try:
            import asyncio

            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        payload = {
            "live_ids": sorted(self.live_ids),
            "context_by_model": dict(sorted(self.context_by_model.items())),
            "probed_ok": sorted(self.probed_ok),
            "dynamic_chains": self.dynamic_chains,
            "saved_at": time.time(),
        }
        if loop is not None:
            # Fire-and-forget: don't block the event loop on disk I/O
            loop.create_task(asyncio.to_thread(save_snapshot, self.snapshot_path, payload))
        else:
            save_snapshot(self.snapshot_path, payload)

    def auto_tokens(self) -> set[str]:
        from potato.routing.auto_router import all_auto_router_ids, is_auto_router_id

        base = {normalize_model_name(t) for t in self.catalog.defaults.auto_mode_model_tokens}
        base |= {
            "potato/auto-coding",
            "potato/best",
            "potato/coding",
            "best",
            "coding",
            "auto-coding",
            "potato/auto-fast",
            "potato/auto-cheap",
            "auto-fast",
            "auto-cheap",
            # OpenRouter / Kilo drop-in aliases
            "openrouter/auto",
            "openrouter/auto-router",
            "kilo/auto",
            "kilo/auto-free",
            "kilo-auto",
            "kilo-auto/frontier",
            "kilo-auto/balanced",
            "kilo-auto/efficient",
            "kilo-auto/free",
        }
        for mid in all_auto_router_ids():
            base.add(normalize_model_name(mid))
        # keep is_auto_router_id in sync for any future aliases
        _ = is_auto_router_id
        return base

    def is_auto(self, model: str | None) -> bool:
        from potato.routing.auto_router import is_auto_router_id

        if model is None or str(model).strip() == "":
            return True
        n = normalize_model_name(model)
        return n in self.auto_tokens() or is_auto_router_id(n)

    def is_alias(self, name: str | None) -> bool:
        n = normalize_model_name(name)
        return n in self.catalog.aliases

    def resolve_alias(self, name: str) -> AliasTarget:
        current = normalize_model_name(name)
        seen: set[str] = set()
        while current in self.catalog.aliases and current not in seen and len(seen) < 10:
            seen.add(current)
            raw = self.catalog.aliases[current]
            target = parse_alias_value(raw)
            if target.kind == "chain":
                return target
            next_name = normalize_model_name(target.value)
            if next_name in self.catalog.aliases and next_name not in seen:
                current = next_name
            else:
                return target
        raw = self.catalog.aliases.get(current, name)
        return parse_alias_value(raw)

    def is_known(self, model_id: str) -> bool:
        return self.resolve_live_id(model_id) is not None

    def resolve_live_id(self, model_id: str, *, include_disabled: bool = False) -> str | None:
        """Map client model id to a namespaced live id when possible.

        Disabled models are treated as unknown unless ``include_disabled``.
        """
        mid = normalize_model_name(model_id)
        if not mid:
            return None

        def _ok(candidate: str) -> bool:
            if candidate not in self.live_ids:
                return False
            if include_disabled:
                return True
            return candidate not in self.disabled_models

        if _ok(mid):
            return mid
        if mid in self.catalog.models and (not self.live_ids or _ok(mid)):
            return mid
        for pid in sorted(self.ladder.provider_ids, key=len, reverse=True):
            cand = f"{pid}/{mid}"
            if _ok(cand):
                return cand
        return None

    def context_length_for(self, model_id: str | None) -> int | None:
        if not model_id:
            return None
        mid = normalize_model_name(model_id)
        return self.context_by_model.get(mid) or self.context_by_model.get(model_id)

    def enrich_model_entry(self, item: dict[str, Any]) -> dict[str, Any]:
        mid = str(item.get("id") or "")
        known = self.context_length_for(mid)
        from_item = extract_context_length(item)
        final = merge_context(from_item, known)
        if final is not None and mid:
            self.context_by_model[mid] = final
        return enrich_model_dict(item, final)

    def _ingest_context_from_api_items(self, items: list[Any]) -> None:
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            mid = str(item["id"])
            got = extract_context_length(item)
            if got is not None:
                self.context_by_model[mid] = merge_context(self.context_by_model.get(mid), got)

    def _ingest_context_from_docs(self) -> None:
        # Match docs to live ids by slug / guessed api id
        by_slug: dict[str, DocModel] = {
            d.slug.lower().replace("_", "-"): d for d in self.doc_models
        }
        for mid in self.live_ids:
            if mid in self.context_by_model:
                continue
            slug = mid.rsplit("/", 1)[-1].lower().replace("_", "-")
            doc = by_slug.get(slug)
            if doc is None:
                continue
            text = f"{doc.description} {doc.slug}"
            got = parse_context_from_text(text)
            if got is not None:
                self.context_by_model[mid] = got

    def model_meta(self, model_id: str):
        return self.catalog.models.get(normalize_model_name(model_id))

    def chain_for_intent(self, intent: str, *, variant: str = "default") -> list[str]:
        if self.catalog.defaults.dynamic_families:
            # Prefer sticky dynamic_chains (precomputed at startup / cache refresh)
            cache_key = intent if variant == "default" else f"{intent}::{variant}"
            if cache_key in self.dynamic_chains and self.dynamic_chains[cache_key]:
                return self._filter_available(list(self.dynamic_chains[cache_key]))
            # Intelligent ladder: strongest available → next → …
            ladder = self.ladder.ladder_for(intent, variant=variant)
            if ladder:
                return self._filter_available(ladder)
            if intent in self.dynamic_chains and self.dynamic_chains[intent]:
                return self._filter_available(list(self.dynamic_chains[intent]))

        # Legacy static fallback behavior
        chain = self.dynamic_chains.get(intent)
        if chain is not None:
            return self._filter_available(list(chain))

        entry = self.catalog.intents.get(intent)
        if entry is None:
            entry = self.catalog.intents.get("coding_agentic")
        if entry is None:
            return []
        return self._filter_available(list(entry.chain))

    def bind_db(self, db: Any) -> None:
        """Attach SQLite and try to restore sticky ranking cache."""
        self._db = db
        if db is None:
            return
        self._load_disabled_models()
        loaded = self.load_rankings_cache()
        if loaded:
            logger.info(
                "using sticky ranking cache (coding head=%s)",
                self.dynamic_chains.get("coding_agentic", [])[:3],
            )
        # Ensure disabled models are out of the sticky pool after restore
        if self.disabled_models:
            self.recompute_rankings(persist=True)
        elif loaded:
            self._sync_chains_from_ladder()

    def active_live_ids(self) -> set[str]:
        """Live catalog minus admin-disabled models (the routable pool)."""
        if not self.disabled_models:
            return set(self.live_ids)
        return {m for m in self.live_ids if m not in self.disabled_models}

    def is_model_enabled(self, model_id: str) -> bool:
        mid = normalize_model_name(model_id)
        return bool(mid) and mid in self.live_ids and mid not in self.disabled_models

    def _canonical_live_id(self, model_id: str) -> str | None:
        """Return the live id matching ``model_id`` ignoring case, or None."""
        mid = normalize_model_name(model_id)
        if not mid:
            return None
        if mid in self.live_ids:
            return mid
        return None

    def original_id(self, model_id: str) -> str:
        """Return the original-case namespaced id for upstream requests.

        Falls back to the input when the model is not in the live pool
        (e.g. passthrough for unknown ids).
        """
        mid = normalize_model_name(model_id)
        return self._original_ids.get(mid, model_id)

    def set_model_enabled(self, model_id: str, enabled: bool) -> dict[str, Any]:
        """Enable/disable a discovered model in the routing pool (persisted)."""
        canonical = self._canonical_live_id(model_id)
        if canonical is None:
            raise ValueError(f"Unknown model '{model_id}' — refresh catalog first")
        mid = canonical
        previous = set(self.disabled_models)
        if enabled:
            self.disabled_models.discard(mid)
        else:
            self.disabled_models.add(mid)
        try:
            self._persist_disabled_models(require_ok=True)
        except Exception:
            self.disabled_models = previous
            raise
        # Rebuild sticky ladders so disabled models leave the pool immediately
        self.recompute_rankings(persist=True)
        return {
            "model_id": mid,
            "enabled": enabled,
            "disabled_count": len(self.disabled_models),
            "active_count": len(self.active_live_ids()),
            "live_count": len(self.live_ids),
        }

    def set_models_enabled(
        self, *, enable: list[str] | None = None, disable: list[str] | None = None
    ) -> dict[str, Any]:
        """Bulk enable/disable. Unknown ids are skipped."""
        previous = set(self.disabled_models)
        changed: list[str] = []
        for mid in enable or []:
            canonical = self._canonical_live_id(mid)
            if canonical and canonical in self.disabled_models:
                self.disabled_models.discard(canonical)
                changed.append(canonical)
        for mid in disable or []:
            canonical = self._canonical_live_id(mid)
            if canonical and canonical not in self.disabled_models:
                self.disabled_models.add(canonical)
                changed.append(canonical)
        if changed:
            try:
                self._persist_disabled_models(require_ok=True)
            except Exception:
                self.disabled_models = previous
                raise
            self.recompute_rankings(persist=True)
        return {
            "changed": changed,
            "disabled_models": sorted(self.disabled_models),
            "active_count": len(self.active_live_ids()),
            "live_count": len(self.live_ids),
        }

    def _persist_disabled_models(self, *, require_ok: bool = False) -> None:
        if self._db is None:
            if require_ok:
                raise RuntimeError("disabled_models not persisted: no sqlite bound")
            return
        try:
            import json

            self._db.set_meta("disabled_models", json.dumps(sorted(self.disabled_models)))
        except Exception:
            logger.exception("persist disabled_models failed")
            if require_ok:
                raise

    def _load_disabled_models(self) -> None:
        if self._db is None:
            return
        try:
            import json

            raw = self._db.get_meta("disabled_models")
            if not raw:
                return
            data = json.loads(raw)
            if isinstance(data, list):
                self.disabled_models = {
                    normalize_model_name(str(x)) for x in data if str(x).strip()
                }
                logger.info("loaded %s disabled model(s) from sqlite", len(self.disabled_models))
        except Exception:
            logger.exception("load disabled_models failed")

    def intent_candidates(self, intent: str) -> list[str]:
        """Every live model capable of serving the given intent. All intents — no coding-only cache.

        Cached: invalidated when live_ids or disabled_models change.
        """
        current_key = (frozenset(self.live_ids), frozenset(self.disabled_models))
        if (
            self._intent_candidates_cache is not None
            and self._intent_candidates_key == current_key
            and intent in self._intent_candidates_cache
        ):
            return list(self._intent_candidates_cache[intent])

        active = self.active_live_ids()
        if not active:
            self._intent_candidates_cache = {}
            self._intent_candidates_key = current_key
            return []
        ladder = getattr(self, "ladder", None)
        if ladder is None:
            result = list(active)
        else:
            result = (
                [m for m in active if ladder.is_coding_capable(m)]
                if intent == "coding_agentic"
                else list(active)
            )

        self._intent_candidates_cache[intent] = result
        self._intent_candidates_key = current_key
        return result

    def coding_candidates(self) -> list[str]:
        """Deprecated shim. Use intent_candidates('coding_agentic')."""
        return self.intent_candidates("coding_agentic")

    def load_rankings_cache(self) -> bool:
        if self._db is None:
            return False
        try:
            data = self._db.get_ranking_cache(self.rankings_cache_key)
        except Exception:
            logger.exception("load rankings cache failed")
            return False
        if not data:
            return False
        ok = self.ladder.import_cache(data, freeze=True)
        if not ok:
            return False
        # Keep live_ids from catalog snapshot if newer set is empty
        cached_live = data.get("live_ids") or []
        if isinstance(cached_live, list) and cached_live and not self.live_ids:
            # Lowercase for membership; preserve original case for upstream round-trip
            for x in cached_live:
                s = str(x)
                self._original_ids[s.lower()] = s
            self.live_ids = {str(x).lower() for x in cached_live}
            self.ladder.live_ids = set(self.active_live_ids())
        self._sync_chains_from_ladder()
        return True

    def persist_rankings_cache(self) -> bool:
        if self._db is None:
            # Fallback to catalog snapshot path sibling
            try:
                path = self.snapshot_path.parent / "rankings_cache.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    __import__("json").dumps(self.ladder.export_cache(), indent=2),
                    encoding="utf-8",
                )
                return True
            except Exception:
                logger.exception("persist rankings json failed")
                return False
        try:
            payload = self.ladder.export_cache()
            payload["live_ids"] = sorted(self.live_ids)
            payload["dynamic_chains"] = {k: list(v) for k, v in self.dynamic_chains.items()}
            self._db.set_ranking_cache(payload, cache_key=self.rankings_cache_key)
            logger.info(
                "ranking cache persisted (coding=%s)",
                payload.get("best_coding", [])[:3],
            )
            return True
        except Exception:
            logger.exception("persist rankings cache failed")
            return False

    def recompute_rankings(self, *, persist: bool = True) -> dict:
        """
        Precompute best open models for all intents and freeze until next refresh.
        Call at startup after catalog load and on admin cache refresh.
        """
        self.ladder.set_docs(self.doc_models)
        self.ladder.provider_ids = set(self.ladder.provider_ids)
        active = self.active_live_ids()
        self.ladder.rebuild(active, freeze=True)
        self._sync_chains_from_ladder()
        if persist:
            self.persist_rankings_cache()
            self._persist_snapshot()
        best = {
            "coding_agentic": self.dynamic_chains.get("coding_agentic", [])[:8],
            "chat_fast": self.dynamic_chains.get("chat_fast", [])[:5],
            "reasoning": self.dynamic_chains.get("reasoning", [])[:5],
            "frozen": self.ladder.frozen,
            "computed_at": self.ladder.computed_at,
            "live_models": len(self.live_ids),
            "active_models": len(active),
            "disabled_models": len(self.disabled_models),
        }
        logger.info("best models precomputed: %s", best)
        return best

    # ------------------------------------------------------------------
    # Intel fetcher + score cache lifecycle (NMK-G804)
    # ------------------------------------------------------------------

    def bind_intel_fetcher(self, fetcher: Any, settings: Any) -> None:
        """Attach IntelFetcher and do a blocking cold-start from disk cache."""
        self._intel_fetcher = fetcher
        self._intel_settings = settings
        self._yaml_scoring_config = self._load_yaml_scoring_config()
        self._sync_score_cache_once()

    def _load_yaml_scoring_config(self) -> dict:
        """Load the scoring section from the YAML catalog file."""
        if self._yaml_path is None or not self._yaml_path.is_file():
            return {}
        try:
            with self._yaml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data
        except Exception:
            logger.exception("yaml scoring config load failed")
            return {}

    def _sync_score_cache_once(self) -> None:
        """Blocking cold-start: build score cache from disk-cached intel bundles."""
        try:
            from potato.catalog.score_cache import ModelScoreCache, recompute

            bundles = (self._intel_fetcher._load_disk_cache() if self._intel_fetcher is not None else {}) or {}
            cache = recompute(
                live_ids=self.active_live_ids(),
                intel_bundles=bundles,
                health=self.health,
                learning=self.learning,
                yaml_cfg=self._yaml_scoring_config,
                provider_ids=getattr(self.ladder, "provider_ids", set()),
            )
            ModelScoreCache.install(cache)
            logger.info("score cache cold-start: %d models from disk", len(cache.scores))
        except Exception:
            logger.exception("score cache cold-start failed")

    async def start_intel_refresh_loop(self) -> None:
        """Background loop: fetch live intel, recompute scores, rebuild ladders."""
        import asyncio

        from potato.catalog.score_cache import ModelScoreCache, recompute

        interval = float(getattr(self._intel_settings, "score_recompute_interval_seconds", 300.0))
        while True:
            try:
                bundles = await self._intel_fetcher.fetch_all()
                cache = recompute(
                    live_ids=self.active_live_ids(),
                    intel_bundles=bundles,
                    health=self.health,
                    learning=self.learning,
                    yaml_cfg=self._yaml_scoring_config,
                    provider_ids=getattr(self.ladder, "provider_ids", set()),
                )
                ModelScoreCache.install(cache)
                self.ladder.rebuild(self.active_live_ids(), freeze=True)
                self._sync_chains_from_ladder()
                logger.info("intel refresh v%d: %d models", cache.version, len(cache.scores))
            except Exception:
                logger.exception("intel refresh loop failed")
            await asyncio.sleep(interval)

    async def trigger_intel_refresh(self) -> dict:
        """Admin endpoint: manual refresh (fire-and-forget)."""
        import asyncio

        from potato.catalog.score_cache import ModelScoreCache

        asyncio.get_running_loop().create_task(self._do_single_refresh())
        curr = ModelScoreCache.current()
        return {"status": "refresh_queued", "current_version": curr.version if curr else 0}

    async def _do_single_refresh(self) -> None:
        from potato.catalog.score_cache import ModelScoreCache, recompute

        try:
            bundles = await self._intel_fetcher.fetch_all()
            cache = recompute(
                live_ids=self.active_live_ids(),
                intel_bundles=bundles,
                health=self.health,
                learning=self.learning,
                yaml_cfg=self._yaml_scoring_config,
                provider_ids=getattr(self.ladder, "provider_ids", set()),
            )
            ModelScoreCache.install(cache)
            self.ladder.rebuild(self.active_live_ids(), freeze=True)
            self._sync_chains_from_ladder()
        except Exception:
            logger.exception("manual intel refresh failed")

    def _sync_chains_from_ladder(self) -> None:
        intents = list(self.catalog.intents.keys()) or [
            "coding_agentic",
            "chat_fast",
            "reasoning",
            "long_horizon",
            "vision",
        ]
        variants_default = {}
        for intent in intents:
            variants_default[intent] = self.ladder.ladder_for(intent, variant="default")
            # Store variant chains for auto-fast / auto-cheap
            self.dynamic_chains[intent] = variants_default[intent]
            self.dynamic_chains[f"{intent}::cheap"] = self.ladder.ladder_for(
                intent, variant="cheap"
            )
            self.dynamic_chains[f"{intent}::fast"] = self.ladder.ladder_for(intent, variant="fast")

    def _rebuild_all_chains(self, *, force: bool = False) -> None:
        """
        Rebuild dynamic chains from ladders.

        Sticky mode: keep precomputed order unless ``force`` or the live catalog
        contains models never ranked (then recompute so new providers join).
        """
        if force:
            self.recompute_rankings(persist=True)
            return
        if self.rankings_sticky and self.ladder.frozen and self.ladder._ladders:
            known: set[str] = set()
            for snap in self.ladder._ladders.values():
                known.update(snap.ladder)
            unseen = set(self.live_ids) - known
            if unseen:
                logger.info(
                    "rankings sticky but %s new model(s) unseen — recompute",
                    len(unseen),
                )
                self.recompute_rankings(persist=True)
                return
            # Only update live set; keep precomputed order (minus disabled)
            if self.disabled_models:
                self.recompute_rankings(persist=True)
                return
            self.ladder.live_ids = set(self.active_live_ids())
            self._sync_chains_from_ladder()
            logger.info(
                "rankings sticky — kept cached ladders (coding head=%s)",
                self.dynamic_chains.get("coding_agentic", [])[:3],
            )
            return
        self.recompute_rankings(persist=True)

    def _filter_available(self, chain: list[str]) -> list[str]:
        # No live catalog yet — keep static YAML chains as-is.
        if not self.live_ids:
            return list(chain)
        active = self.active_live_ids()
        filtered = [m for m in chain if m in active]
        for m in chain:
            if m not in self.live_ids:
                logger.warning("catalog: skipping unavailable model id %s", m)
            elif m in self.disabled_models:
                logger.debug("catalog: skipping admin-disabled model id %s", m)
        if not filtered and self.strict_catalog:
            raise RuntimeError("strict_catalog: no models available for chain")
        # Never fail-open to disabled/unavailable models (empty pool → []).
        return filtered

    def health_reorder(
        self, chain: list[str], *, intent: str = "coding_agentic", variant: str = "default"
    ) -> list[str]:
        """
        Always algorithmically rank: intelligence × live speed × health.

        Sticky ladder provides intelligence prior; every request re-scores
        candidates so the fastest *responding* strong models lead.
        """
        from potato.routing.optimizer import optimize_chain

        return optimize_chain(chain, self, intent=intent, variant=variant, max_n=None)

    def record_outcome(
        self,
        model: str,
        key_id: str | None,
        success: bool,
        latency: float | None = None,
        status_code: int | None = None,
        unavailable: bool = False,
        tokens: int | None = None,
        *,
        intent: str | None = None,
        empty_reply: bool = False,
        had_tools: bool = False,
        tool_ok: bool | None = None,
    ) -> None:
        self.health.record_outcome(
            model,
            key_id=key_id,
            success=success,
            latency=latency,
            status_code=status_code,
            unavailable=unavailable,
            tokens=tokens,
        )
        if success:
            self.probed_ok.add(model)
        if intent:
            self.learning.record(
                intent=intent,
                model_id=model,
                success=success,
                unavailable=unavailable,
                empty_reply=empty_reply,
                had_tools=had_tools,
                tool_ok=tool_ok,
            )
            # Sticky rankings: do NOT rebuild ladder on every outcome.
            # Online learning still records; order refreshes only on cache refresh.
            if not self.rankings_sticky or not self.ladder.frozen:
                self._sync_chains_from_ladder()

    def _join_docs_to_ids(self) -> set[str]:
        """Map doc slugs/publishers onto live API ids."""
        if not self.doc_models or not self.live_ids:
            return set(self.live_ids)

        by_slug: dict[str, str] = {}
        for mid in self.live_ids:
            slug = mid.split("/", 1)[-1].lower()
            by_slug.setdefault(slug, mid)

        matched: set[str] = set()
        for doc in self.doc_models:
            slug = doc.slug.lower().replace("_", "-")
            # try exact slug match against api model name
            for live_slug, mid in by_slug.items():
                if live_slug == slug or live_slug.replace("_", "-") == slug:
                    matched.add(mid)
                    break
            guess = doc.api_id_guess
            if guess and guess in self.live_ids:
                matched.add(guess)
        # Always keep full live set for resolution; docs enrich descriptions only
        return set(self.live_ids)

    async def refresh_from_upstream(
        self,
        upstream: UpstreamClient,
        *,
        fetch_docs: bool = True,
        run_probes: bool = True,
    ) -> bool:
        api_ok = False
        try:
            status, body, _headers, _key = await upstream.request_json("GET", "/models")
            if status >= 400:
                logger.warning("catalog refresh failed: HTTP %s", status)
            else:
                ids: set[str] = set()
                data = body.get("data") if isinstance(body, dict) else None
                if isinstance(data, list):
                    self._ingest_context_from_api_items(data)
                    for item in data:
                        if isinstance(item, dict) and item.get("id"):
                            original = str(item["id"]).strip()
                            ids.add(original.lower())
                            self._original_ids[original.lower()] = original
                if ids:
                    self.live_ids = ids
                    api_ok = True
                    logger.info("catalog API refresh ok — %s live model(s)", len(ids))
        except Exception:
            logger.exception("catalog API refresh error")

        if fetch_docs:
            try:
                docs = await fetch_models_md(self.docs_url)
                if docs and self.enrich_doc_details:
                    docs = await enrich_publishers(docs, limit=30)
                if docs:
                    self.doc_models = docs
                    self.last_docs_ok = True
                else:
                    self.last_docs_ok = False
            except Exception:
                logger.exception("docs refresh failed")
                self.last_docs_ok = False

        if not api_ok and not self.live_ids:
            # fail-safe: keep snapshot
            self.last_refresh_ok = False
            logger.warning("catalog refresh degraded — using snapshot/live cache")
            if self.dynamic_chains:
                return False
            return False

        self._join_docs_to_ids()
        self._ingest_context_from_docs()
        self._rebuild_all_chains()

        if run_probes and self.probe_budget.remaining() > 0:
            # Probe only chain heads we care about (anti-ban)
            candidates: list[str] = []
            for intent in ("coding_agentic", "chat_fast"):
                for mid in self.dynamic_chains.get(intent, [])[:2]:
                    if mid not in candidates:
                        candidates.append(mid)
            if candidates:
                results = await probe_models(upstream, candidates, self.probe_budget)
                for mid, st in results.items():
                    if st in {"ok", "rate_limited"}:
                        self.probed_ok.add(mid)
                    elif st == "unavailable":
                        self.health.record_outcome(
                            mid, success=False, status_code=404, unavailable=True
                        )
                self._rebuild_all_chains()

        self.last_refresh_at = time.monotonic()
        self.last_refresh_ok = api_ok or bool(self.live_ids)
        self._persist_snapshot()
        return self.last_refresh_ok

    async def refresh_single_provider(
        self, provider_id: str, hub: Any, *, recompute_rankings: bool = True
    ) -> bool:
        """Fetch /models from one provider and merge into live_ids immediately."""
        from potato.catalog.providers import namespace_model

        rt = hub.runtimes.get(provider_id.lower())
        if rt is None or not rt.config.enabled:
            logger.warning("provider %s has no active runtime — skip fetch", provider_id)
            return False

        try:
            status, body, _h, _k = await rt.upstream.request_json("GET", "/models")
            if status >= 400 or not isinstance(body, dict):
                logger.warning("provider %s /models → HTTP %s", provider_id, status)
                return False
            data = body.get("data")
            if not isinstance(data, list):
                return False
            pid = provider_id.lower()
            new_ids: set[str] = set()
            namespaced_items: list[dict] = []
            whitelist = rt.config.model_whitelist
            blacklist = rt.config.model_blacklist
            for item in data:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                upstream_id = str(item["id"])
                # Per-provider model whitelist/blacklist (NMK-103)
                low_uid = upstream_id.lower()
                if blacklist and any(b.lower() in low_uid for b in blacklist):
                    continue
                if whitelist and not any(w.lower() in low_uid for w in whitelist):
                    continue
                ns = namespace_model(pid, upstream_id)
                new_ids.add(ns)
                namespaced_items.append({**item, "id": ns})
            self._ingest_context_from_api_items(namespaced_items)
            # Lowercase for membership; preserve original case for upstream round-trip
            new_ids_lower: set[str] = set()
            for ns in new_ids:
                self._original_ids[ns.lower()] = ns
                new_ids_lower.add(ns.lower())
            self.live_ids |= new_ids_lower
            # Sync ladder provider ids so scoring/split uses the new provider
            self.ladder.provider_ids = set(m.split("/", 1)[0] for m in self.live_ids)
            logger.info(
                "provider %s imported %s model(s) into live pool (total=%s)",
                pid,
                len(new_ids),
                len(self.live_ids),
            )
        except Exception:
            logger.exception("provider %s catalog fetch failed", provider_id)
            return False

        if recompute_rankings:
            self._rebuild_all_chains()
        self._persist_snapshot()
        return True

    async def refresh_from_hub(
        self,
        hub: Any,
        *,
        fetch_docs: bool = True,
        run_probes: bool = True,
        recompute_rankings: bool | None = None,
    ) -> bool:
        """
        Refresh live catalog from all enabled OpenAI-compatible providers.

        ``recompute_rankings``:
          - True  → force precompute + persist best-model cache
          - False → keep sticky rankings (only update live_ids)
          - None  → recompute only if rankings are empty / not sticky
        """
        from potato.catalog.providers import namespace_model

        self.ladder.provider_ids = set(hub.provider_ids)
        merged: set[str] = set()
        failed_pids: set[str] = set()
        any_ok = False

        # Snapshot runtimes to avoid RuntimeError on concurrent admin mutations
        for pid, rt in list(hub.runtimes.items()):
            if not rt.config.enabled:
                continue
            if not rt.config.resolved_keys():
                logger.warning("provider %s has no API keys — skip catalog fetch", pid)
                continue
            try:
                status, body, _h, _k = await rt.upstream.request_json("GET", "/models")
                if status >= 400 or not isinstance(body, dict):
                    logger.warning("provider %s /models → HTTP %s", pid, status)
                    continue
                data = body.get("data")
                if not isinstance(data, list):
                    continue
                namespaced_items: list[dict] = []
                for item in data:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    upstream_id = str(item["id"])
                    # Per-provider model whitelist/blacklist (NMK-103)
                    whitelist = rt.config.model_whitelist
                    blacklist = rt.config.model_blacklist
                    low_uid = upstream_id.lower()
                    if blacklist and any(b.lower() in low_uid for b in blacklist):
                        continue
                    if whitelist and not any(w.lower() in low_uid for w in whitelist):
                        continue
                    ns = namespace_model(pid, upstream_id)
                    merged.add(ns)
                    namespaced_items.append({**item, "id": ns})
                self._ingest_context_from_api_items(namespaced_items)
                any_ok = True
                logger.info("provider %s catalog ok — %s model(s)", pid, len(namespaced_items))
            except Exception:
                logger.exception("provider %s catalog refresh failed", pid)
                failed_pids.add(pid)

        if merged:
            # Lowercase for O(1) membership; preserve original case for upstream round-trip
            self._original_ids = {m.lower(): m for m in merged}
            # Retain models from providers that failed this refresh (avoid eviction)
            if failed_pids:
                retained = {m for m in self.live_ids if m.split("/", 1)[0] in failed_pids}
                merged_lower = {m.lower() for m in merged}
                for m in retained:
                    if m.lower() not in merged_lower:
                        self._original_ids[m.lower()] = m
                merged = {m.lower() for m in merged} | retained
            self.live_ids = merged
        elif not self.live_ids:
            self.last_refresh_ok = False
            logger.warning("hub catalog refresh degraded — no live models")
            return False

        if fetch_docs:
            try:
                docs = await fetch_models_md(self.docs_url)
                if docs and self.enrich_doc_details:
                    docs = await enrich_publishers(docs, limit=30)
                if docs:
                    self.doc_models = docs
                    self.last_docs_ok = True
            except Exception:
                logger.exception("docs refresh failed")
                self.last_docs_ok = False

        self._join_docs_to_ids()
        self._ingest_context_from_docs()

        force_rank = recompute_rankings
        if force_rank is None:
            force_rank = (
                (not self.rankings_sticky) or (not self.ladder.frozen) or (not self.ladder._ladders)
            )
        if force_rank or self.disabled_models:
            self.recompute_rankings(persist=True)
        else:
            self.ladder.live_ids = set(self.active_live_ids())
            self._sync_chains_from_ladder()

        if run_probes and self.probe_budget.remaining() > 0:
            candidates: list[str] = []
            for intent in ("coding_agentic", "chat_fast"):
                for mid in self.dynamic_chains.get(intent, [])[:2]:
                    if mid not in candidates:
                        candidates.append(mid)
            for mid in candidates:
                if self.probe_budget.remaining() <= 0:
                    break
                try:
                    client, _pid, upstream_mid = hub.client_for_model(mid)
                    results = await probe_models(client, [upstream_mid], self.probe_budget)
                    st = results.get(upstream_mid)
                    if st in {"ok", "rate_limited"}:
                        self.probed_ok.add(mid)
                    elif st == "unavailable":
                        self.health.record_outcome(
                            mid, success=False, status_code=404, unavailable=True
                        )
                except Exception:
                    logger.debug("probe failed for %s", mid, exc_info=True)
            # Probes update health only; sticky rankings stay frozen

        self.last_refresh_at = time.monotonic()
        self.last_refresh_ok = any_ok or bool(self.live_ids)
        self._persist_snapshot()
        return self.last_refresh_ok

    def snapshot(self) -> dict[str, Any]:
        age = None
        if self.last_refresh_at is not None:
            age = round(time.monotonic() - self.last_refresh_at, 1)
        active = self.active_live_ids()
        return {
            "yaml_version": self.catalog.version,
            "yaml_updated": self.catalog.updated,
            "yaml_path": str(self._yaml_path) if self._yaml_path else None,
            "live_model_count": len(self.live_ids),
            "active_model_count": len(active),
            "disabled_model_count": len(self.disabled_models),
            "live_ids": sorted(self.live_ids),
            "disabled_models": sorted(self.disabled_models),
            "context_known_count": len(self.context_by_model),
            "docs_count": len(self.doc_models),
            "docs_ok": self.last_docs_ok,
            "probed_ok_count": len(self.probed_ok),
            "probe_budget_remaining": self.probe_budget.remaining(),
            "last_refresh_age_s": age,
            "last_refresh_ok": self.last_refresh_ok,
            "rankings_sticky": self.rankings_sticky,
            "rankings_frozen": self.ladder.frozen,
            "rankings_computed_at": self.ladder.computed_at,
            "best_coding": list(self.dynamic_chains.get("coding_agentic", [])[:10]),
            "best_chat": list(self.dynamic_chains.get("chat_fast", [])[:8]),
            "dynamic_families": self.catalog.defaults.dynamic_families,
            "families": self.catalog.families.model_dump(),
            "ladders": self.ladder.snapshot(),
            "learning": self.learning.snapshot(),
            "dynamic_chains": dict(self.dynamic_chains),
            "intents": {
                name: {
                    "description": ic.description,
                    "chain": self.chain_for_intent(name),
                    "primary_family": ic.primary_family,
                }
                for name, ic in self.catalog.intents.items()
            },
            "aliases": dict(self.catalog.aliases),
            "health": self.health.snapshot(),
        }

    def max_advertised_context(self) -> int:
        """Honest context ceiling for synthetic potato/* model cards.

        Returns the largest known context window across the active live pool,
        so the advertised number never exceeds what the best available upstream
        can actually handle — clients won't oversend and 400 upstream.
        Falls back to 131072 (128K) when no live context is known yet.
        """
        try:
            active = self.active_live_ids()
            known = [
                v
                for mid, v in self.context_by_model.items()
                if mid in active and isinstance(v, int) and v > 0
            ]
        except Exception:
            known = []
        return max(known) if known else 131072

    def synthetic_auto_model(self) -> dict[str, Any]:
        ctx = self.max_advertised_context()
        return {
            "id": "potato/auto",
            "object": "model",
            "created": 0,
            "owned_by": "potato",
            "permission": [],
            "root": "potato/auto",
            "parent": None,
            "context_length": ctx,
            "max_model_len": ctx,
            "supports_cache": True,
        }

    def synthetic_auto_models(self) -> list[dict[str, Any]]:
        """All virtual models Cursor / OpenAI clients can pick from /v1/models.

        Includes OpenRouter- and Kilo-compatible auto-router ids so clients that
        hard-code ``openrouter/auto`` or ``kilo-auto/*`` work drop-in.

        Also includes Anthropic model-name proxies so Claude Code CLI (which
        validates model names client-side against /v1/models) can pass its own
        model ids through to the gateway unchanged.
        """
        from potato.routing.auto_router import all_auto_router_ids

        base = {
            "object": "model",
            "created": 0,
            "owned_by": "potato",
            "permission": [],
            "parent": None,
            "context_length": self.max_advertised_context(),
            "max_model_len": self.max_advertised_context(),
            "supports_tools": True,
            "supports_vision": True,
            "supports_cache": True,
        }
        owned = {
            "openrouter/auto": "openrouter",
            "kilo/auto": "kilo",
            "kilo-auto/frontier": "kilo",
            "kilo-auto/balanced": "kilo",
            "kilo-auto/efficient": "kilo",
            "kilo-auto/free": "kilo",
        }
        out: list[dict[str, Any]] = []
        for mid in all_auto_router_ids():
            out.append(
                {
                    **base,
                    "id": mid,
                    "root": mid,
                    "owned_by": owned.get(mid, "potato"),
                }
            )

        # Anthropic / Claude Code CLI proxy names ─ these appear in /v1/models
        # so the CLI's client-side model validation passes.  The alias system
        # in models.yaml routes them to the correct intent chain.
        anthropic_base = {
            **base,
            "owned_by": "anthropic",
        }
        seen_ids: set[str] = set()
        for alias_name in self.catalog.aliases:
            if alias_name.startswith("claude-"):
                entry = {
                    **anthropic_base,
                    "id": alias_name,
                    "root": alias_name,
                }
                out.append(entry)
                seen_ids.add(alias_name)
                # Claude Code CLI appends context-window suffixes like [1m]
                # for long-context models.  Emit suffixed variants so the
                # CLI's client-side exact-match validation always passes.
                suffixed = f"{alias_name}[1m]"
                if suffixed not in seen_ids:
                    out.append({**entry, "id": suffixed, "root": alias_name})
                    seen_ids.add(suffixed)

        return out

