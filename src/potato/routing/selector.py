"""Resolve client model field → ordered NIM model chain.

OpenRouter / Kilo parity: ``openrouter/auto``, ``kilo/auto``, ``kilo-auto/*``
trigger prompt-aware selection with optional session model pin + plugins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from potato.catalog.aliases import looks_like_nim_id, normalize_model_name
from potato.routing.auto_router import (
    AutoRouterOptions,
    build_intent_aware_pool,
    filter_chain,
    is_auto_router_id,
    pin_model_first,
    resolve_auto_tier,
    sticky_fits_intent_pool,
    tier_to_variant,
)
from potato.routing.intents import Intent, IntentResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from potato.catalog.registry import ModelRegistry
    from potato.config import Settings

RouteMode = Literal[
    "auto",
    "alias",
    "alias_model",
    "passthrough",
    "passthrough_with_fallback",
    "unknown_alias_as_auto",
    "disabled",
]


@dataclass
class RouteDecision:
    chain: list[str]
    mode: RouteMode
    intent: Intent
    rule_id: str
    requested_model: str | None
    # OpenRouter-style metadata
    auto_tier: str | None = None
    variant: str = "default"
    sticky_model: str | None = None
    # Sticky/explicit head that _chain must keep first unless unhealthy
    pinned_head: str | None = None
    allowed_models: list[str] = field(default_factory=list)
    # Estimated prompt tokens for context-length filtering (T13)
    estimated_tokens: int | None = None
    # 12-D LinUCB feature vector captured at classify time, used to score the
    # chain (rank_chain_with_rl) and to record feedback (record_feedback).
    feature_vector: list[float] | None = None


class ModelSelector:
    def __init__(
        self,
        registry: ModelRegistry,
        settings: Settings,
        preferences: Any | None = None,
        model_ladders: Any | None = None,
        rl_engine: Any | None = None,
        model_pools: Any | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.preferences = preferences
        self.model_ladders = model_ladders
        self.rl_engine = rl_engine
        self.model_pools = model_pools

    def rank_chain_with_rl(
        self, chain: list[str], feature_vector: list[float] | None = None
    ) -> list[str]:
        if not self.rl_engine or not chain or not feature_vector or len(chain) <= 1:
            return chain

        scored: list[tuple[float, str]] = []
        for idx, mid in enumerate(chain):
            rl_score, _, _ = self.rl_engine.score(mid, feature_vector)
            # Multiplicative boost bounded in [0.5, 2.0] so baseline capability is preserved
            boost = max(0.5, min(2.0, 1.0 + rl_score))
            # Preserve base candidate order via subtle tie-breaker (anti-
            # celebrity: keep the prior weak so RL can actually reorder)
            position_weight = 1.0 / (1.0 + 0.01 * idx)
            composite = boost * position_weight
            scored.append((composite, mid))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [mid for _, mid in scored]

    def _max_n_for_intent(self, intent: str) -> int:
        """Per-intent fallback cap from config (replaces coding_max_fallbacks)."""
        limits = getattr(self.settings, "intent_max_fallbacks", {}) or {}
        default = int(getattr(self.settings, "max_model_fallbacks", 10) or 10)
        return int(limits.get(intent, default))

    def resolve(
        self,
        model_field: str | None,
        intent_result: IntentResult,
        *,
        routing_disabled: bool = False,
        auto_opts: AutoRouterOptions | None = None,
        preferred_model: str | None = None,
    ) -> RouteDecision:
        intent = intent_result.intent
        if intent == Intent.UNKNOWN:
            intent = Intent.CODING_AGENTIC

        raw = normalize_model_name(model_field)
        if not raw and self.settings.default_model:
            raw = normalize_model_name(self.settings.default_model)

        opts = auto_opts or AutoRouterOptions()
        tier = opts.tier or resolve_auto_tier(raw if raw else model_field)
        variant = tier_to_variant(tier) if tier else "default"

        # Force coding intent for coding / frontier tiers (agent + architecture)
        if (
            tier in ("coding", "frontier")
            and intent not in (Intent.EMBEDDINGS, Intent.VISION)
            and (tier == "coding" or intent == Intent.CHAT_FAST)
        ):
            # frontier keeps reasoning/long_horizon when classified
            intent = Intent.CODING_AGENTIC

        intent_key = intent.value

        if routing_disabled:
            chain = [raw] if raw else []
            if not chain and self.settings.default_model:
                # ponytail: unreachable — raw is populated from default_model at
                # line 91-92, so `not raw` implies no default_model. Defensive.
                chain = [self.settings.default_model]  # pragma: no cover
            return RouteDecision(
                chain=chain or ["auto"],
                mode="disabled",
                intent=intent,
                rule_id=intent_result.rule_id,
                requested_model=model_field,
                auto_tier=tier,
                variant=variant,
            )

        requested_live = self.registry.resolve_live_id(raw, include_disabled=True)
        if requested_live in self.registry.disabled_models:
            raise ValueError("model_disabled")

        # Custom model group check (e.g. potato/coding, potato/fast, or custom admin groups)
        if self.model_ladders is not None and (raw or model_field):
            target_id = raw if raw else str(model_field or "")
            lad = self.model_ladders.get(target_id)
            if lad is None and model_field:
                lad = self.model_ladders.get(str(model_field))
            if lad is not None and lad.chain:
                # 1. Closed candidate pool defined by administrator for this model group
                group_candidates = list(lad.chain)

                # 2. Dynamic scoring & RL re-ranking over the group candidate pool
                # Evaluates quality scores, intent affinity, live health & RL bandit metrics
                scored_group_chain = self.registry.health_reorder(
                    group_candidates, intent=intent_key, variant=variant
                )

                # 3. Finalize chain (apply allowed filters, user prefs, and fallback bounds)
                chain = self._finalize_chain(
                    scored_group_chain,
                    intent_key=intent_key,
                    variant=variant,
                    free_only=tier == "free",
                    allowed=opts.allowed_models,
                    preferred_model=preferred_model if tier else None,
                    models_fallback=opts.models_fallback,
                    is_custom_ladder=True,
                )
                if chain:
                    return RouteDecision(
                        chain=chain,
                        mode="passthrough_with_fallback",
                        intent=intent,
                        rule_id=f"custom_ladder:{lad.model_id}",
                        requested_model=model_field,
                        auto_tier=tier,
                        variant=variant,
                        allowed_models=list(opts.allowed_models),
                    )
                # ponytail: custom ladder existed but _finalize_chain emptied it
                # (chain models not live, admin-disabled, or pool-gated out).
                # Fall back to potato/auto rather than the generic intent pool —
                # the admin defined this ladder for a reason; if it can't serve,
                # let the auto-router pick a working model instead of pulling
                # random models from the intent pool.
                logger.warning(
                    "custom_ladder %s chain emptied by finalize (live/disabled/pool) — "
                    "falling back to potato/auto for %s",
                    lad.model_id,
                    target_id,
                )
                # Recurse as potato/auto so the auto-router serves the request.
                # Preserve intent + variant so the auto-router still respects the
                # classified intent rather than re-classifying.
                return self.resolve(
                    "potato/auto",
                    intent_result,
                    auto_opts=auto_opts,
                    preferred_model=preferred_model,
                )

        # User preferences first
        if self.preferences is not None and self.preferences.has_preference(intent_key):
            pref = self.preferences.get(intent_key)
            if pref is not None and pref.chain:
                mode: RouteMode = "passthrough" if pref.strict else "passthrough_with_fallback"
                chain = list(pref.chain)
                if not pref.strict:
                    siblings = self.registry.chain_for_intent(intent_key, variant=variant)
                    chain = chain + [m for m in siblings if m not in chain]
                chain = self._finalize_chain(
                    chain,
                    intent_key=intent_key,
                    variant=variant,
                    free_only=tier == "free",
                    allowed=opts.allowed_models,
                    preferred_model=preferred_model if tier else None,
                    models_fallback=opts.models_fallback,
                )
                return RouteDecision(
                    chain=chain,
                    mode=mode,
                    intent=intent,
                    rule_id=f"user_pref:{intent_key}",
                    requested_model=model_field,
                    auto_tier=tier,
                    variant=variant,
                    sticky_model=preferred_model,
                    pinned_head=preferred_model,
                    allowed_models=list(opts.allowed_models),
                )

        if intent == Intent.EMBEDDINGS:
            chain = self.registry.chain_for_intent("embeddings", variant=variant)
            # Explicit enabled embedding model must lead (not be swallowed by chain head)
            if raw and not self.registry.is_auto(raw):
                resolved = self.registry.resolve_live_id(raw)
                if resolved:
                    chain = [resolved] + [m for m in chain if m != resolved]
                elif not chain and looks_like_nim_id(raw):
                    # Unknown passthrough only when not admin-disabled
                    disabled_hit = self.registry.resolve_live_id(raw, include_disabled=True)
                    if disabled_hit in self.registry.disabled_models:
                        # ponytail: unreachable — a disabled model that's in
                        # live_ids is rejected at line 123-125 before reaching
                        # here; one not in live_ids yields disabled_hit=None.
                        raise ValueError("model_disabled")  # pragma: no cover
                    chain = [raw]
            elif not chain and raw and looks_like_nim_id(raw):
                chain = [raw]
            chain = self.registry.health_reorder(chain, intent=intent_key, variant=variant)
            # Keep explicit pin first after health reorder
            if raw and not self.registry.is_auto(raw):
                resolved = self.registry.resolve_live_id(raw)
                if resolved:
                    chain = pin_model_first(chain, resolved)
            return RouteDecision(
                chain=chain,
                mode="auto" if self.registry.is_auto(raw) or not raw else "passthrough",
                intent=intent,
                rule_id=intent_result.rule_id,
                requested_model=model_field,
                auto_tier=tier,
                variant=variant,
                pinned_head=self.registry.resolve_live_id(raw) if raw else None,
            )

        if intent == Intent.VISION:
            chain = self.registry.chain_for_intent("vision", variant=variant)
            chain = self.registry.health_reorder(chain, intent="vision", variant=variant)
            if not chain:
                # Never route image requests to alphabetical text models (T19)
                raise ValueError("no_vision_model")
            return RouteDecision(
                chain=chain,
                mode="auto",
                intent=intent,
                rule_id=intent_result.rule_id,
                requested_model=model_field,
                auto_tier=tier,
                variant=variant,
                pinned_head=preferred_model,
                sticky_model=preferred_model,
            )

        # Auto router (OpenRouter/Kilo/Potato virtual models)
        is_auto = (
            raw == ""
            or self.registry.is_auto(raw)
            or is_auto_router_id(model_field)
            or is_auto_router_id(raw)
        )
        if is_auto:
            return self._resolve_auto(
                intent=intent,
                intent_result=intent_result,
                intent_key=intent_key,
                variant=variant,
                tier=tier,
                model_field=model_field,
                opts=opts,
                preferred_model=preferred_model,
            )

        if self.registry.is_alias(raw):
            target = self.registry.resolve_alias(raw)
            if target.kind == "chain":
                chain = self.registry.chain_for_intent(target.value, variant=variant)
                chain = self._finalize_chain(
                    chain,
                    intent_key=target.value,
                    variant=variant,
                    free_only=False,
                    allowed=opts.allowed_models,
                    preferred_model=None,
                    models_fallback=opts.models_fallback,
                )
                return RouteDecision(
                    chain=chain,
                    mode="alias",
                    intent=Intent(target.value)
                    if target.value in {i.value for i in Intent}
                    else intent,
                    rule_id=intent_result.rule_id,
                    requested_model=model_field,
                    variant=variant,
                )
            # alias → concrete model
            target_model = (
                self.registry.resolve_live_id(target.value, include_disabled=True) or target.value
            )
            if target_model in self.registry.disabled_models:
                raise ValueError("model_disabled")
            chain = [target_model]
            if self.settings.enable_fallback_on_explicit:
                siblings = self.registry.chain_for_intent(intent_key, variant=variant)
                chain = chain + [m for m in siblings if m != target_model]
            # Health-reorder once for both branches; pin the alias target first
            # so the client never sees a surprise head model (F-08 invariant).
            optimized = self.registry.health_reorder(chain, intent=intent_key, variant=variant)
            head = target_model
            rest = [m for m in optimized if m != head]
            return RouteDecision(
                chain=[head] + rest,
                mode="alias_model",
                intent=intent,
                rule_id=intent_result.rule_id,
                requested_model=model_field,
                variant=variant,
                pinned_head=head,
            )

        if self.registry.is_known(raw) or looks_like_nim_id(raw):
            resolved = self.registry.resolve_live_id(raw)
            if resolved is None:
                # Disabled or unknown: reject disabled explicitly; unknown NIM ids
                # may still passthrough if they are not in the disabled set.
                disabled_hit = self.registry.resolve_live_id(raw, include_disabled=True)
                if disabled_hit in self.registry.disabled_models:
                    # ponytail: unreachable — disabled live models are rejected
                    # at line 123-125; non-live models yield disabled_hit=None.
                    raise ValueError("model_disabled")  # pragma: no cover
                if not looks_like_nim_id(raw):
                    # Fall through to auto below
                    # ponytail: unreachable — entering this branch via is_known
                    # requires resolve_live_id to return non-None (catalog model
                    # with empty live_ids), so resolved is never None here.
                    resolved = None  # pragma: no cover
                else:
                    resolved = raw
            if resolved is not None:
                if self.settings.enable_fallback_on_explicit:
                    siblings = self.registry.chain_for_intent(intent_key, variant=variant)
                    bare = resolved.split("/")[-1] if "/" in resolved else resolved
                    horizontals = [
                        m
                        for m in siblings
                        if (m.split("/")[-1] if "/" in m else m) == bare and m != resolved
                    ]
                    rest = [m for m in siblings if m != resolved and m not in horizontals]
                    rest_opt = self.registry.health_reorder(
                        rest, intent=intent_key, variant=variant
                    )
                    chain = (
                        [resolved]
                        + [m for m in horizontals if m != resolved]
                        + [m for m in rest_opt if m != resolved and m not in horizontals]
                    )
                    mode: RouteMode = "passthrough_with_fallback"
                else:
                    chain = [resolved]
                    mode = "passthrough"
                # For coding, rank candidates but keep the requested model pinned
                # so _chain / sticky affinity is not silently discarded (F-08).
                if intent_key == "coding_agentic":
                    chain = self.registry.health_reorder(chain, intent=intent_key, variant=variant)
                    chain = pin_model_first(chain, resolved)
                return RouteDecision(
                    chain=chain,
                    mode=mode,
                    intent=intent,
                    rule_id=intent_result.rule_id,
                    requested_model=model_field,
                    variant=variant,
                    pinned_head=resolved,
                )

        # Unknown non-NIM string → treat as auto (Cursor defaults, etc.)
        return self._resolve_auto(
            intent=intent,
            intent_result=intent_result,
            intent_key=intent_key,
            variant=variant,
            tier=tier,
            model_field=model_field,
            opts=opts,
            preferred_model=preferred_model if tier else None,
            mode="unknown_alias_as_auto",
        )

    def _resolve_auto(
        self,
        *,
        intent: Intent,
        intent_result: IntentResult,
        intent_key: str,
        variant: str,
        tier: str | None,
        model_field: str | None,
        opts: AutoRouterOptions,
        preferred_model: str | None,
        mode: RouteMode = "auto",
    ) -> RouteDecision:
        """Always produce an intent-aligned chain for potato/auto (and aliases).

        Guarantees: if any active model exists and hard filters allow it, the
        chain is non-empty. Intent-ladder models lead (highest score first);
        ALL remaining live models fill the tail so the best available model is
        always reachable — nothing is silently excluded from the candidate set.
        """
        free_only = tier == "free"
        max_n = self._max_n_for_intent(intent_key)

        # Build a full-catalog pool: intent-ranked models lead, then ALL live
        # models follow (score-sorted). include_all_live=True ensures potato/auto
        # has the complete catalog as its candidate set before finalization.
        chain = build_intent_aware_pool(
            self.registry,
            primary_intent=intent_key,
            variant=variant,
            max_n=max_n,
            include_related=True,
            include_all_live=True,
        )

        # Soft sticky: only pin when the model fits this intent (or low confidence)
        pin = preferred_model
        if pin and not sticky_fits_intent_pool(
            pin,
            chain,
            confidence=float(intent_result.confidence or 0.0),
            force_intent=intent_result.rule_id
            in {"tools_present", "agent_fingerprint", "agent_header", "forced_header"},
        ):
            pin = None

        chain = self._finalize_chain(
            chain,
            intent_key=intent_key,
            variant=variant,
            free_only=free_only,
            allowed=opts.allowed_models,
            preferred_model=pin,
            models_fallback=opts.models_fallback,
        )

        # Cap to max_n after finalization (not before) so health_reorder gets the
        # full scored pool and the actual best model is at the head.
        if chain and max_n and len(chain) > max_n:
            chain = chain[:max_n]

        # Hard guarantee for auto: never return empty when the live pool has models
        if not chain:
            chain = build_intent_aware_pool(
                self.registry,
                primary_intent=intent_key,
                variant=variant,
                max_n=max_n,
                include_related=True,
                include_all_live=True,
            )
            chain = filter_chain(
                chain,
                allowed_models=opts.allowed_models or None,
                free_only=free_only,
            )
            if not chain and not opts.allowed_models and not free_only:
                # Absolute last resort — any active model, no free/allowed constraints
                # ponytail: unreachable — when live_ids is empty, _filter_available
                # fail-opens to the static YAML chain, so the finalized chain is
                # never empty here. Kept as a defensive guarantee against future
                # changes to _filter_available.
                try:  # pragma: no cover
                    chain = sorted(self.registry.active_live_ids())[:max_n]
                except Exception:  # pragma: no cover
                    chain = []

        return RouteDecision(
            chain=chain,
            mode=mode,
            intent=intent,
            rule_id=intent_result.rule_id,
            requested_model=model_field,
            auto_tier=tier or ("balanced" if mode == "auto" else tier),
            variant=variant,
            sticky_model=pin,
            pinned_head=pin,
            allowed_models=list(opts.allowed_models),
        )

    def _finalize_chain(
        self,
        chain: list[str],
        *,
        intent_key: str,
        variant: str,
        free_only: bool,
        allowed: list[str],
        preferred_model: str | None,
        models_fallback: list[str],
        is_custom_ladder: bool = False,
    ) -> list[str]:
        # Custom ladders: only keep models that are actually live in the
        # registry. If none survive, the caller (resolve) falls back to
        # potato/auto instead of the generic intent pool.
        if is_custom_ladder and self.registry.live_ids:
            chain = [
                m for m in chain
                if (self.registry.resolve_live_id(m) or m) in self.registry.live_ids
                or m in self.registry.live_ids
            ]
        # OpenRouter models[] as extra fallback candidates
        if models_fallback:
            for m in models_fallback:
                discovered = self.registry.resolve_live_id(m, include_disabled=True)
                if discovered in self.registry.disabled_models:
                    continue
                resolved = self.registry.resolve_live_id(m) or m
                if resolved not in chain:
                    chain = chain + [resolved]

        chain = self.registry.health_reorder(chain, intent=intent_key, variant=variant)
        chain = filter_chain(chain, allowed_models=allowed or None, free_only=free_only)
        # Drop admin-disabled models from every finalized route chain
        if self.registry.live_ids and self.registry.disabled_models:
            chain = [
                m
                for m in chain
                if (self.registry.resolve_live_id(m, include_disabled=True) or m)
                not in self.registry.disabled_models
            ]

        # Granular Model Pool & Intent Gating Filtering
        if self.model_pools and chain:
            is_auto = (
                False
                if is_custom_ladder
                else bool(variant in ("auto", "default") or intent_key == "auto")
            )
            chain = [
                m
                for m in chain
                if self.model_pools.is_allowed(m, intent_key, is_auto_router=is_auto)
            ]
        # Session model pin (OpenRouter sticky routing) — pin once after ranking
        preferred_live = (
            self.registry.resolve_live_id(preferred_model, include_disabled=True)
            if preferred_model
            else None
        )
        if preferred_model and preferred_live not in self.registry.disabled_models:
            chain = pin_model_first(chain, preferred_model)
        return chain
