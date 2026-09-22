"""Online learning from routing outcomes — Thompson Sampling + UCB1.

Tracks per-(intent, model) success/failure counts for:
  - Thompson Sampling: Beta(α, β) distribution for Bayesian quality estimation
  - UCB1: request counts for exploration bonus calculation
  - Legacy EWMA: backward-compatible score_delta() for callers
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelLearningStats:
    successes: int = 0
    failures: int = 0
    empty_replies: int = 0
    tool_ok: int = 0
    tool_fail: int = 0
    unavailable: int = 0
    # EWMA quality in [-1, 1]: + good, - bad (legacy, kept for compat)
    ewma_quality: float = 0.0
    last_updated: float = 0.0
    # Total requests routed to this model for this intent (UCB1)
    total_requests: int = 0

    @property
    def alpha(self) -> float:
        """Beta distribution α parameter (successes + optimistic prior)."""
        return self.successes + 1.0

    @property
    def beta_param(self) -> float:
        """Beta distribution β parameter (failures + prior)."""
        return self.failures + 1.0

    def score_delta(self) -> float:
        """
        Legacy soft adjustment.  Kept for backward compat with callers that
        still read score_delta.  New code uses thompson_params / ucb instead.
        """
        total = self.successes + self.failures
        delta = self.ewma_quality * 12.0
        if self.unavailable >= 2:
            delta -= 25.0
        if self.tool_fail > self.tool_ok and self.tool_fail >= 2:
            delta -= 15.0
        if self.empty_replies >= 3:
            delta -= 10.0
        if total >= 5 and self.successes / max(total, 1) > 0.9:
            delta += 5.0
        return max(-40.0, min(25.0, delta))


@dataclass
class LearningStore:
    """Persisted per-(intent, model) learning signals with Thompson + UCB1."""

    path: Path = field(default_factory=lambda: Path(".potato/learning.json"))
    save_debounce_seconds: float = 10.0
    _data: dict[str, dict[str, ModelLearningStats]] = field(default_factory=dict)
    # Per-intent total request counters for UCB1
    _intent_totals: dict[str, int] = field(default_factory=dict)
    _dirty: bool = False
    _last_save_at: float = 0.0
    # Epoch counter: bumped on every record(); save clears only if epoch matches
    _epoch: int = 0
    _save_scheduled: bool = False
    _bg_tasks: set[Any] = field(default_factory=set)

    def _key(self, intent: str, model_id: str) -> tuple[str, str]:
        return intent, model_id

    def stats(self, intent: str, model_id: str) -> ModelLearningStats:
        bucket = self._data.setdefault(intent, {})
        if model_id not in bucket:
            bucket[model_id] = ModelLearningStats()
        return bucket[model_id]

    def record(
        self,
        *,
        intent: str,
        model_id: str,
        success: bool,
        unavailable: bool = False,
        empty_reply: bool = False,
        had_tools: bool = False,
        tool_ok: bool | None = None,
    ) -> None:
        s = self.stats(intent, model_id)
        s.last_updated = time.time()
        s.total_requests += 1
        self._intent_totals[intent] = self._intent_totals.get(intent, 0) + 1

        if unavailable:
            s.unavailable += 1
            s.failures += 1
            s.ewma_quality = 0.7 * s.ewma_quality + 0.3 * (-1.0)
        elif success:
            s.successes += 1
            q = 1.0
            if empty_reply:
                s.empty_replies += 1
                q = -0.3
            if had_tools:
                if tool_ok:
                    s.tool_ok += 1
                    q = 1.0
                elif tool_ok is False:
                    s.tool_fail += 1
                    q = -0.8
            s.ewma_quality = 0.7 * s.ewma_quality + 0.3 * q
        else:
            s.failures += 1
            s.ewma_quality = 0.7 * s.ewma_quality + 0.3 * (-1.0)
        self._dirty = True
        self._epoch += 1
        self.save_if_due()

    # ------------------------------------------------------------------
    # Thompson Sampling interface
    # ------------------------------------------------------------------

    def thompson_params(self, intent: str, model_id: str) -> tuple[float, float]:
        """Return (α, β) for Beta distribution — used by Thompson Sampling."""
        s = self.stats(intent, model_id)
        return s.alpha, s.beta_param

    # ------------------------------------------------------------------
    # UCB1 interface
    # ------------------------------------------------------------------

    def total_requests(self, intent: str) -> int:
        """Total requests across all models for this intent."""
        return self._intent_totals.get(intent, 0)

    def model_requests(self, intent: str, model_id: str) -> int:
        """Total requests routed to a specific model for this intent."""
        if intent not in self._data or model_id not in self._data[intent]:
            return 0
        return self._data[intent][model_id].total_requests

    # ------------------------------------------------------------------
    # Legacy interface (backward compat)
    # ------------------------------------------------------------------

    def score_delta(self, intent: str, model_id: str) -> float:
        if intent not in self._data or model_id not in self._data[intent]:
            return 0.0
        return self._data[intent][model_id].score_delta()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for intent, models in (raw.get("intents") or {}).items():
                for mid, d in models.items():
                    st = ModelLearningStats(
                        successes=int(d.get("successes", 0)),
                        failures=int(d.get("failures", 0)),
                        empty_replies=int(d.get("empty_replies", 0)),
                        tool_ok=int(d.get("tool_ok", 0)),
                        tool_fail=int(d.get("tool_fail", 0)),
                        unavailable=int(d.get("unavailable", 0)),
                        ewma_quality=float(d.get("ewma_quality", 0.0)),
                        last_updated=float(d.get("last_updated", 0.0)),
                        total_requests=int(d.get("total_requests", 0)),
                    )
                    self._data.setdefault(intent, {})[mid] = st
            # Rebuild intent totals from loaded data
            for intent, models in self._data.items():
                self._intent_totals[intent] = sum(s.total_requests for s in models.values())
            logger.info("loaded learning store (%s intents)", len(self._data))
        except Exception:
            logger.exception("failed to load learning store")

    def save_if_due(self, *, force: bool = False) -> None:
        if not self._dirty and not force:
            return
        now = time.time()
        if (
            not force
            and self._last_save_at
            and now - self._last_save_at < self.save_debounce_seconds
        ):
            return
        # Prefer off-loop disk I/O when called from an async request path (T11)
        import os
        if "PYTEST_CURRENT_TEST" in os.environ:
            self.save()
            return

        try:
            import asyncio

            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and not force:
            if getattr(self, "_save_scheduled", False):
                return
            self._save_scheduled = True

            async def _save_bg() -> None:
                try:
                    await asyncio.to_thread(self.save)
                finally:
                    self._save_scheduled = False

            task = loop.create_task(_save_bg())
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            return
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot epoch before writing; if record() bumps it during write,
        # _dirty stays True and the next save will pick up the new data.
        epoch_at_start = self._epoch
        # Shallow-copy mutable dicts to avoid mutation during iteration
        data_snapshot = dict(self._data)
        totals_snapshot = dict(self._intent_totals)
        payload = {
            "saved_at": time.time(),
            "intent_totals": totals_snapshot,
            "intents": {
                intent: {
                    mid: {
                        "successes": st.successes,
                        "failures": st.failures,
                        "empty_replies": st.empty_replies,
                        "tool_ok": st.tool_ok,
                        "tool_fail": st.tool_fail,
                        "unavailable": st.unavailable,
                        "ewma_quality": round(st.ewma_quality, 4),
                        "last_updated": st.last_updated,
                        "score_delta": round(st.score_delta(), 2),
                        "total_requests": st.total_requests,
                        "thompson_alpha": round(st.alpha, 1),
                        "thompson_beta": round(st.beta_param, 1),
                    }
                    for mid, st in models.items()
                }
                for intent, models in data_snapshot.items()
            },
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        # Only clear dirty if no new records arrived during the write
        if self._epoch == epoch_at_start:
            self._dirty = False
        self._last_save_at = time.time()

    async def flush(self) -> None:
        """Await all pending background save tasks."""
        import asyncio
        tasks = [t for t in list(self._bg_tasks) if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._dirty:
            await asyncio.to_thread(self.save)

    def snapshot(self) -> dict:
        return {
            intent: {
                mid: {
                    "score_delta": round(st.score_delta(), 2),
                    "ewma_quality": round(st.ewma_quality, 3),
                    "successes": st.successes,
                    "failures": st.failures,
                    "total_requests": st.total_requests,
                    "thompson_alpha": round(st.alpha, 1),
                    "thompson_beta": round(st.beta_param, 1),
                }
                for mid, st in models.items()
            }
            for intent, models in self._data.items()
        }
