"""Atomic, versionable, periodically-recomputed model score cache.

Replaces hardcoded QUALITY_TIERS regex and INTENT_AFFINITY dicts.
Quality scores computed from live internet sources (benchmarks, ELOs, TPS),
cached atomically, refreshed periodically, and corrected continuously by
Thompson Sampling from real routing outcomes.

The YAML config is a robust fallback, not the primary source of truth.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, ClassVar

from potato.catalog.intel_fetcher import IntelBundle

logger = logging.getLogger(__name__)

PARAM_RE = re.compile(r"(?:^|[^a-z0-9])(\d{1,4})b(?:[^a-z0-9]|$)", re.I)


@dataclass
class ModelScore:
    model_id: str
    quality: float  # 0-100 composite
    intent_affinity: dict[str, float]  # intent -> multiplicative factor
    modalities: frozenset[str]  # "text","tools","vision","reasoning","embeddings"
    context_k: float  # context window in K tokens
    measured_tps: float  # 0 = unknown
    provider_id: str
    sources: list[str]
    computed_at: float


@dataclass
class ModelScoreCache:
    scores: dict[str, ModelScore]
    version: int
    computed_at: float
    live_pool: frozenset[str]

    _current: ClassVar[ModelScoreCache | None] = None

    @classmethod
    def current(cls) -> ModelScoreCache | None:
        return cls._current

    @classmethod
    def install(cls, new: ModelScoreCache) -> None:
        """Atomic reference swap. Read path never needs a lock."""
        cls._current = new

    def get(self, model_id: str) -> ModelScore | None:
        return self.scores.get(model_id)


# ---------------------------------------------------------------------------
# Quality computation (no regex tables)
# ---------------------------------------------------------------------------


def _compute_quality(bundle: IntelBundle | None, slug: str, yaml_cfg: dict) -> float:
    cfg = yaml_cfg.get("scoring", {})
    wc = cfg.get("quality_signal_weights", {})
    bounds = cfg.get("quality_bounds", {"min": 10.0, "max": 100.0})
    arena_base = float(cfg.get("arena_elo_base", 800))
    arena_scale = float(cfg.get("arena_elo_scale", 8.0))

    w_aa = float(wc.get("aa_intelligence", 0.40))
    w_hf = float(wc.get("hf_leaderboard", 0.30))
    w_elo = float(wc.get("arena_elo", 0.20))
    w_par = float(wc.get("param_estimate", 0.10))

    signals: list[tuple[float, float]] = []

    if bundle:
        if bundle.aa_intelligence_idx is not None:
            signals.append((float(bundle.aa_intelligence_idx), w_aa))
        hf = _hf_composite(bundle)
        if hf is not None:
            signals.append((hf, w_hf))
        if bundle.arena_elo is not None:
            elo_norm = min(100.0, max(0.0, (bundle.arena_elo - arena_base) / arena_scale))
            signals.append((elo_norm, w_elo))
        if w_par > 0 and bundle.param_b is not None and bundle.param_b > 0:
            # Param-size signal (same log-scale formula as the slug fallback,
            # clamped so unknown-param models never exceed known SOTA).
            p = float(bundle.param_b)
            signals.append((min(84.0, max(50.0, 58.0 + 6.0 * math.log2(p / 7.0))), w_par))

    if signals:
        total_w = sum(w for _, w in signals)
        quality = sum(v * w for v, w in signals) / max(1e-5, total_w)
    else:
        quality = _slug_quality_fallback(slug, cfg)

    return max(float(bounds["min"]), min(float(bounds["max"]), quality))


def _hf_composite(bundle: IntelBundle) -> float | None:
    parts = [x for x in (bundle.hf_mmlu, bundle.hf_humaneval) if x is not None]
    return sum(parts) / len(parts) if parts else None


def _slug_quality_fallback(slug: str, cfg: dict) -> float:
    s = str(slug or "").strip().lower()

    # Real-world benchmark quality priors (LMSYS Chatbot Arena ELO / Artificial Analysis Quality Index)
    # Tier 1: SOTA Reasoning & Agentic Coding (97.0 - 99.0)
    if any(k in s for k in ("claude-3-7", "claude-3.7", "o3-mini", "o3", "o1", "gpt-4.5")):
        return 98.5
    if any(k in s for k in ("claude-3-5-sonnet", "claude-3.5-sonnet", "gpt-4o")):
        return 97.5
    if any(k in s for k in ("glm-5.2", "deepseek-r1", "deepseek-v3", "qwen3.5-397b")):
        return 96.5

    # Tier 2: High-Capability Frontier & Agentic Coding Specialists (89.5 - 94.0)
    if any(k in s for k in ("glm-5.1", "gemini-2.0-pro", "gemini-1.5-pro", "claude-3-opus")):
        return 94.0
    if any(k in s for k in ("glm-5", "qwen2.5-coder-32b", "qwen2.5-coder", "qwen3.5-122b", "llama-3.3-70b")):
        return 93.0
    if any(k in s for k in ("nemotron-3-ultra", "nemotron-3-ultra-550b", "nemotron-4-340b")):
        return 91.5
    if any(k in s for k in ("nemotron-3-super", "nemotron-3-super-120b", "mistral-large")):
        return 89.5

    # Tier 3: Capable Mid-Size & Fast Flash Models (85.0 - 87.0)
    if any(k in s for k in ("gpt-4o-mini", "claude-3-5-haiku", "gemini-2.0-flash", "gemini-1.5-flash")):
        return 86.5
    if any(k in s for k in ("glm-4-plus", "glm-4", "step-3.7", "step-3", "minimax-m3")):
        return 85.0

    # 2. Config keyword floors
    kws = cfg.get("quality_floor_keywords", {}) if isinstance(cfg, dict) else {}
    for kw, score in kws.items():
        if kw != "_default" and kw in s:
            return float(score)

    # 3. Extract parameter size estimate log scale (capped at 84.0 so unknown param models never exceed known SOTA)
    m = PARAM_RE.search(s)
    if m:
        with suppress(ValueError, TypeError):
            p = float(m.group(1))
            if p > 0:
                return min(84.0, max(50.0, 58.0 + 6.0 * math.log2(p / 7.0)))

    return float(kws.get("_default", 65.0))


# ---------------------------------------------------------------------------
# Modality detection (no regex tables)
# ---------------------------------------------------------------------------


def _compute_modalities(
    bundle: IntelBundle | None, model_id: str, yaml_cfg: dict
) -> frozenset[str]:
    modalities: set[str] = set()
    slug = _normalize_slug_from_id(model_id)
    mid = model_id.lower()
    hints = (yaml_cfg.get("scoring") or {}).get("capability_hints", {})

    # Hard-exclude non-chat models
    if any(_glob_match(mid, p) for p in hints.get("exclude_chat_patterns", [])):
        if "embed" in mid or "rerank" in mid:
            return frozenset({"embeddings"})
        return frozenset()

    modalities.add("text")

    if bundle:
        if bundle.supports_tools is True:
            modalities.add("tools")
        if bundle.supports_vision is True:
            modalities.add("vision")
        if bundle.supports_reasoning is True:
            modalities.add("reasoning")
        if bundle.supports_embeddings is True:
            modalities.add("embeddings")
            modalities.discard("text")

    # Fill unknowns from YAML hints
    if (
        "tools" not in modalities
        and (not bundle or bundle.supports_tools is None)
        and any(_glob_match(slug, p) for p in hints.get("tools_true_patterns", []))
    ):
        modalities.add("tools")
    if (
        "vision" not in modalities
        and (not bundle or bundle.supports_vision is None)
        and any(_glob_match(slug, p) for p in hints.get("vision_true_patterns", []))
    ):
        modalities.add("vision")
    if (
        "reasoning" not in modalities
        and (not bundle or bundle.supports_reasoning is None)
        and any(_glob_match(slug, p) for p in hints.get("reasoning_true_patterns", []))
    ):
        modalities.add("reasoning")

    return frozenset(modalities)


# ---------------------------------------------------------------------------
# Intent affinity computation (no static dict)
# ---------------------------------------------------------------------------


def _compute_intent_affinity(
    bundle: IntelBundle | None,
    modalities: frozenset[str],
    quality: float,
    learning: Any,
    model_id: str,
    yaml_cfg: dict,
) -> dict[str, float]:
    intents = [
        "coding_agentic",
        "reasoning",
        "long_horizon",
        "chat_fast",
        "vision",
        "embeddings",
    ]
    # Built-in defaults so cold-start (empty YAML) still produces sane affinities.
    _DEFAULT_DELTAS: dict[str, dict[str, float]] = {
        "tools_confirmed_true": {"coding_agentic": 0.25, "reasoning": 0.25},
        "tools_confirmed_false": {"coding_agentic": -0.80, "reasoning": -0.80},
        "vision_confirmed_true": {"vision": 0.30},
        "vision_confirmed_false": {"vision": -0.95},
        "reasoning_confirmed_true": {"reasoning": 0.30, "coding_agentic": 0.30},
        "long_context": {"long_horizon": 0.25},
        "short_context": {"long_horizon": -0.30},
        "small_param": {"chat_fast": 0.20},
        "frontier_param": {"coding_agentic": 0.30, "reasoning": 0.30},
        "embed_confirmed_true": {"embeddings": 0.30},
        "embed_confirmed_false": {"embeddings": -0.95},
    }
    yaml_deltas = (yaml_cfg.get("scoring") or {}).get("capability_affinity_deltas", {})
    # YAML overrides defaults per-key when present
    deltas = {
        k: {**_DEFAULT_DELTAS.get(k, {}), **(yaml_deltas.get(k) or {})}
        for k in set(_DEFAULT_DELTAS) | set(yaml_deltas)
    }
    affinity: dict[str, float] = {}

    for intent in intents:
        base = 1.0

        # Hard-exclude mismatched modalities
        if intent == "vision" and "vision" not in modalities:
            affinity[intent] = 0.05
            continue
        if intent == "embeddings" and "embeddings" not in modalities:
            affinity[intent] = 0.05
            continue
        if intent not in ("vision", "embeddings") and "text" not in modalities:
            affinity[intent] = 0.05
            continue

        # Apply capability deltas from YAML (all in one place, data-driven)
        if "tools" in modalities:
            base += float((deltas.get("tools_confirmed_true") or {}).get(intent, 0.0))
        elif bundle and bundle.supports_tools is False:
            base += float((deltas.get("tools_confirmed_false") or {}).get(intent, 0.0))

        if "vision" in modalities:
            base += float((deltas.get("vision_confirmed_true") or {}).get(intent, 0.0))
        elif bundle and bundle.supports_vision is False:
            base += float((deltas.get("vision_confirmed_false") or {}).get(intent, 0.0))

        if "reasoning" in modalities:
            base += float((deltas.get("reasoning_confirmed_true") or {}).get(intent, 0.0))

        ctx_k = (bundle.context_length or 0) / 1000 if bundle and bundle.context_length else 0
        if ctx_k >= 100:
            base += float((deltas.get("long_context") or {}).get(intent, 0.0))
        elif 0 < ctx_k < 16:
            base += float((deltas.get("short_context") or {}).get(intent, 0.0))

        param_b = bundle.param_b if bundle else None
        if param_b is not None:
            if param_b < 10:
                base += float((deltas.get("small_param") or {}).get(intent, 0.0))
            elif param_b > 100:
                base += float((deltas.get("frontier_param") or {}).get(intent, 0.0))

        # Primary family preference from YAML policy
        fam_policy = yaml_cfg.get("families") or {}
        chat_prim = str(fam_policy.get("chat_primary") or "").lower()
        coding_prim = str(fam_policy.get("coding_primary") or "").lower()
        mid_lower = model_id.lower()
        if intent == "chat_fast" and chat_prim and chat_prim in mid_lower:
            base += 0.25
        if intent == "coding_agentic" and coding_prim and coding_prim in mid_lower:
            base += 0.25

        # Quality tier boost for hard intents
        if (intent in ("coding_agentic", "reasoning") and quality >= 90) or (
            intent == "chat_fast" and quality < 70
        ):
            base *= 1.10

        # Thompson posterior from real routing outcomes (intent-specific)
        try:
            alpha, beta_p = learning.thompson_params(intent, model_id)
            posterior_mean = alpha / (alpha + beta_p)
            posterior_factor = 0.5 + 1.0 * posterior_mean
        except Exception:
            posterior_factor = 1.0

        affinity[intent] = round(max(0.05, base * posterior_factor), 4)

    return affinity


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def recompute(
    live_ids: set[str],
    intel_bundles: dict[str, IntelBundle],
    health: Any,
    learning: Any,
    yaml_cfg: dict,
    provider_ids: set[str] | None = None,
) -> ModelScoreCache:
    """Pure function: given inputs, produce a new ModelScoreCache. No side effects."""
    provider_ids = provider_ids or set()
    scoring_cfg = yaml_cfg.get("scoring", {})
    speed_priors = scoring_cfg.get("provider_speed_prior", {})
    scores: dict[str, ModelScore] = {}

    for model_id in live_ids:
        slug = _normalize_slug_from_id(model_id)
        bundle = intel_bundles.get(slug) or _fuzzy_match_bundle(slug, intel_bundles)

        quality = _compute_quality(bundle, slug, yaml_cfg)
        modalities = _compute_modalities(bundle, model_id, yaml_cfg)
        affinity = _compute_intent_affinity(
            bundle, modalities, quality, learning, model_id, yaml_cfg
        )

        # Provider ID: split from namespaced model_id
        pid = model_id.split("/")[0] if "/" in model_id else "nim"

        # TPS: prefer measured EWMA, then AA data, then provider prior
        tps = 0.0
        h = getattr(health, "_by_model", {}).get(model_id)
        if h and getattr(h, "ewma_tok_per_s", 0) > 0:
            tps = h.ewma_tok_per_s
        elif bundle and bundle.aa_tps:
            tps = bundle.aa_tps
        else:
            prior = float(speed_priors.get(pid) or speed_priors.get("_default") or 1.0)
            tps = prior * 40.0  # normalize: prior 1.0 = 40 TPS baseline

        scores[model_id] = ModelScore(
            model_id=model_id,
            quality=quality,
            intent_affinity=affinity,
            modalities=modalities,
            context_k=(bundle.context_length or 0) / 1000 if bundle else 0,
            measured_tps=tps,
            provider_id=pid,
            sources=bundle.sources if bundle else ["param_estimate"],
            computed_at=time.time(),
        )

    prev = ModelScoreCache.current()
    cache = ModelScoreCache(
        scores=scores,
        version=(prev.version + 1) if prev else 1,
        computed_at=time.time(),
        live_pool=frozenset(live_ids),
    )
    logger.info(
        "score_cache recomputed v%d: %d models, sources=%s",
        cache.version,
        len(scores),
        sorted({s for ms in scores.values() for s in ms.sources}),
    )
    return cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_slug_from_id(model_id: str) -> str:
    bare = str(model_id or "").strip().lower().rsplit("/", 1)[-1]
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", bare)


def _fuzzy_match_bundle(slug: str, bundles: dict[str, IntelBundle]) -> IntelBundle | None:
    """
    Deterministic fuzzy match for model bundles.
    Ensures model family AND parameter size tokens (e.g., 8b, 70b, 8x7b, 8x22b) match.
    """
    slug_tokens = set(re.findall(r"[a-z0-9]+", slug.lower()))
    slug_nums = set(re.findall(r"\d+[bkm]?", slug.lower()))

    best_bundle: IntelBundle | None = None
    best_score = 0.0

    for bslug, b in sorted(bundles.items(), key=lambda item: item[0]):
        btokens = set(re.findall(r"[a-z0-9]+", bslug.lower()))
        bnums = set(re.findall(r"\d+[bkm]?", bslug.lower()))

        # Parameter sizes/numbers must not conflict (e.g. 7b != 70b, 8x7b != 8x22b)
        if slug_nums and bnums and slug_nums != bnums:
            continue

        intersection = slug_tokens & btokens
        union = slug_tokens | btokens
        if not union:
            continue
        jaccard = len(intersection) / len(union)

        if jaccard >= 0.75 and jaccard > best_score:
            best_score = jaccard
            best_bundle = b

    return best_bundle


def _glob_match(s: str, pattern: str) -> bool:
    return fnmatch.fnmatch(s, pattern)
