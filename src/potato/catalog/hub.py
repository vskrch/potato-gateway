"""Per-provider upstream clients + key pools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from potato.balancer import KeyPool
from potato.catalog.providers import (
    ProviderConfig,
    ProviderStore,
    namespace_model,
    split_provider_model,
)
from potato.config import Settings
from potato.safety.circuit_breaker import ProviderCircuitBreaker
from potato.upstream import UpstreamClient

logger = logging.getLogger(__name__)


@dataclass
class ProviderRuntime:
    config: ProviderConfig
    pool: KeyPool
    upstream: UpstreamClient


class ProviderHub:
    """
    OpenRouter-style hub: many OpenAI-compatible backends.
    `default` is the nim (or first enabled) upstream for legacy call sites.
    """

    def __init__(self, store: ProviderStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.runtimes: dict[str, ProviderRuntime] = {}
        self.circuit_breaker = ProviderCircuitBreaker()

    @property
    def provider_ids(self) -> set[str]:
        return self.store.provider_ids()

    @property
    def default(self) -> UpstreamClient:
        if "nim" in self.runtimes and self.runtimes["nim"].config.enabled:
            return self.runtimes["nim"].upstream
        for rt in self.runtimes.values():
            if rt.config.enabled:
                return rt.upstream
        # Empty nim client may exist for fail-closed
        if "nim" in self.runtimes:
            return self.runtimes["nim"].upstream
        raise RuntimeError("No providers configured")

    @property
    def default_pool(self) -> KeyPool:
        if "nim" in self.runtimes:
            return self.runtimes["nim"].pool
        for rt in self.runtimes.values():
            return rt.pool
        raise RuntimeError("No providers configured")

    def gate_capacity(self) -> int:
        """Sum of key-slots across enabled provider runtimes (F-09)."""
        total = 0
        for rt in self.runtimes.values():
            if not rt.config.enabled:
                continue
            per_key = max(1, int(rt.config.max_in_flight_per_key or 1))
            total += len(rt.pool) * per_key
        return total

    async def start(self) -> None:
        for cfg in self.store.providers.values():
            await self._ensure_runtime(cfg)

    async def stop(self) -> None:
        for rt in self.runtimes.values():
            await rt.upstream.stop()
        self.runtimes.clear()

    async def _ensure_runtime(self, cfg: ProviderConfig) -> ProviderRuntime | None:
        pid = cfg.id.lower()
        keys = cfg.resolved_keys()
        if pid in self.runtimes:
            # recreate if base_url changed — simple: always rebuild on upsert
            await self.runtimes[pid].upstream.stop()
            del self.runtimes[pid]

        if not cfg.enabled:
            return None
        if cfg.api_style != "openai":
            logger.warning(
                "provider %s api_style=%s skipped (phase 1 openai only)",
                pid,
                cfg.api_style,
            )
            return None

        if not keys:
            logger.warning(
                "provider %s has no API keys — skipping runtime (add keys via dashboard or config)",
                pid,
            )
            return None

        pool = KeyPool(
            api_keys=keys,
            rpm_limit=max(1.0, cfg.rpm_limit * self.settings.nim_rpm_safety_factor),
            cooldown_seconds=self.settings.nim_cooldown_seconds,
            rpd_limit=cfg.rpd_limit,
            max_in_flight_per_key=cfg.max_in_flight_per_key,
            auth_fail_threshold=self.settings.auth_fail_threshold,
            auth_quarantine_seconds=self.settings.auth_quarantine_seconds,
            sticky_boost=self.settings.sticky_boost,
        )
        upstream = UpstreamClient(
            base_url=cfg.base_url or self.settings.nim_base_url,
            pool=pool,
            timeout=self.settings.upstream_timeout,
            connect_timeout=getattr(self.settings, "upstream_connect_timeout_seconds", 5.0),
            user_agent=self.settings.upstream_user_agent,
            proxy_url=self.settings.egress_proxy_url(),
            retry_backoff_base=self.settings.retry_backoff_base_seconds,
            retry_backoff_cap=self.settings.retry_backoff_cap_seconds,
        )
        await upstream.start()
        rt = ProviderRuntime(config=cfg, pool=pool, upstream=upstream)
        self.runtimes[pid] = rt
        logger.info(
            "provider %s ready — base=%s keys=%s",
            pid,
            cfg.base_url,
            len(keys),
        )
        return rt

    async def upsert_provider(self, cfg: ProviderConfig, registry: Any = None) -> dict[str, Any]:
        self.store.upsert(cfg)
        await self._ensure_runtime(self.store.providers[cfg.id.lower()])
        self._notify_capacity_change()
        # Immediate /models fetch into live pool (NMK-101)
        if registry is not None and self.has_runtime(cfg.id):
            try:
                await registry.refresh_single_provider(cfg.id, self, recompute_rankings=True)
            except Exception:
                logger.exception("immediate model fetch failed for provider %s", cfg.id)
        return self.store.providers[cfg.id.lower()].mask()

    def _notify_capacity_change(self) -> None:
        resize = getattr(self, "_on_capacity_change", None)
        if callable(resize):
            try:
                resize()
            except Exception:
                logger.debug("gate resize callback failed", exc_info=True)

    async def remove_provider(self, provider_id: str) -> bool:
        ok = self.store.remove(provider_id)
        pid = provider_id.lower()
        if pid in self.runtimes:
            cfg = self.store.providers.get(pid)
            if cfg is None or not cfg.enabled:
                await self.runtimes[pid].upstream.stop()
                del self.runtimes[pid]
            else:
                await self._ensure_runtime(cfg)
        self._notify_capacity_change()
        return ok

    def has_runtime(self, provider_id: str) -> bool:
        """True when provider has an active client with keys."""
        rt = self.runtimes.get(provider_id.lower())
        return rt is not None and rt.config.enabled and bool(rt.config.resolved_keys())

    def active_provider_ids(self) -> set[str]:
        return {pid for pid in self.runtimes if self.has_runtime(pid)}

    def client_for_model(self, model_id: str) -> tuple[UpstreamClient, str, str]:
        """
        Returns (upstream, provider_id, upstream_model_id).

        Never silently sends a namespaced model to the wrong provider — that
        was a production footgun (e.g. groq/... routed to NIM → 404 cascade).
        Raises RuntimeError when the owning provider has no active runtime so
        FallbackExecutor can advance to the next chain model.
        """
        pid, upstream_mid = split_provider_model(
            model_id, self.provider_ids, default_provider="nim"
        )
        # Circuit breaker: skip providers in open state (NMK-401)
        if not self.circuit_breaker.allow(pid):
            # Last resort: if ALL active providers are open, force-allow this one
            # (better to try and fail than to not try at all)
            active_pids = self.active_provider_ids() or self.provider_ids
            if not self.circuit_breaker.any_closed(active_pids):
                logger.warning("all active provider circuits open — force-allowing %s", pid)
                self.circuit_breaker.force_allow(pid)
            else:
                raise RuntimeError(
                    f"provider '{pid}' circuit is open — skipping model '{model_id}'"
                )
        rt = self.runtimes.get(pid)
        if rt is not None and rt.config.enabled and rt.config.resolved_keys():
            return rt.upstream, pid, upstream_mid

        # Config states — don't trip the circuit breaker (only HTTP/transport
        # failures should do that). FallbackExecutor will advance to next model.
        if rt is None or not rt.config.enabled:
            raise RuntimeError(f"provider '{pid}' is not available for model '{model_id}'")
        raise RuntimeError(f"provider '{pid}' has no API keys for model '{model_id}'")

    def namespace(self, provider_id: str, upstream_model_id: str) -> str:
        return namespace_model(provider_id, upstream_model_id)
