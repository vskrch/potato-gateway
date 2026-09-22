"""Multi-key pool with RPM limiting and response-rate aware selection."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from potato.safety.budgets import utc_day_key

logger = logging.getLogger(__name__)


@dataclass
class KeyStats:
    """Live stats for one upstream API key."""

    key_id: str
    api_key: str
    # Sliding window of request timestamps (monotonic)
    request_times: deque[float] = field(default_factory=deque)
    # EWMA of latency in seconds (successes only)
    ewma_latency: float = 1.0
    success_count: int = 0
    error_count: int = 0
    rate_limit_hits: int = 0
    # Monotonic time until which this key is cooling down (after 429)
    cooldown_until: float = 0.0
    in_flight: int = 0
    # Account-safety extensions
    auth_failures: int = 0
    quarantined_until: float = 0.0
    daily_count: int = 0
    daily_window_start: str = ""  # UTC YYYY-MM-DD
    last_request_at: float = 0.0

    def mask(self) -> str:
        k = self.api_key
        if len(k) <= 12:
            return "***"
        return f"{k[:8]}...{k[-4:]}"


class KeyPool:
    """
    Rotates across NVIDIA NIM API keys.

    Selection goals:
    1. Never pick a key that would exceed RPM (with safety factor).
    2. Prefer keys with lower recent latency / fewer failures (response-rate aware).
    3. Spread load (shuffle + weighted pick) so no single account is hammered.
    4. Skip quarantined / over daily budget / over max in-flight keys.
    """

    def __init__(
        self,
        api_keys: list[str],
        rpm_limit: float = 36.0,
        cooldown_seconds: float = 60.0,
        window_seconds: float = 60.0,
        *,
        rpd_limit: int = 2000,
        max_in_flight_per_key: int = 3,
        auth_fail_threshold: int = 2,
        auth_quarantine_seconds: float = 3600.0,
        sticky_boost: float = 3.0,
    ) -> None:
        self.rpm_limit = rpm_limit
        self.cooldown_seconds = cooldown_seconds
        self.window_seconds = window_seconds
        self.rpd_limit = rpd_limit
        self.max_in_flight_per_key = max_in_flight_per_key
        self.auth_fail_threshold = auth_fail_threshold
        self.auth_quarantine_seconds = auth_quarantine_seconds
        self.sticky_boost = sticky_boost
        self._lock = asyncio.Lock()
        # Empty pool is allowed so the app can boot and return clear 503s;
        # acquire() fails closed until NIM_API_KEYS is configured.
        self._keys: list[KeyStats] = [
            KeyStats(key_id=f"key-{i}", api_key=k) for i, k in enumerate(api_keys)
        ]
        if not self._keys:
            logger.error("KeyPool has zero NIM keys — all upstream requests will fail closed")
        self._rr = 0

    def __len__(self) -> int:
        return len(self._keys)

    def _prune(self, stats: KeyStats, now: float) -> None:
        cutoff = now - self.window_seconds
        while stats.request_times and stats.request_times[0] < cutoff:
            stats.request_times.popleft()

    def _roll_daily(self, stats: KeyStats) -> None:
        today = utc_day_key(datetime.now(UTC))
        if stats.daily_window_start != today:
            stats.daily_window_start = today
            stats.daily_count = 0

    def _rpm_used(self, stats: KeyStats, now: float) -> int:
        self._prune(stats, now)
        return len(stats.request_times)

    def _is_available(self, stats: KeyStats, now: float) -> bool:
        if now < stats.cooldown_until:
            return False
        if now < stats.quarantined_until:
            return False
        self._roll_daily(stats)
        if self.rpd_limit > 0 and stats.daily_count >= self.rpd_limit:
            return False
        if stats.in_flight >= self.max_in_flight_per_key:
            return False
        return self._rpm_used(stats, now) < self.rpm_limit

    def _score(self, stats: KeyStats, now: float, *, preferred_key_id: str | None = None) -> float:
        used = self._rpm_used(stats, now)
        headroom = max(0.0, self.rpm_limit - used - stats.in_flight * 0.5)
        latency = max(0.05, stats.ewma_latency)
        latency_score = 1.0 / latency

        total = stats.success_count + stats.error_count
        success_rate = (stats.success_count / total) if total else 1.0
        concurrency_penalty = 1.0 / (1.0 + stats.in_flight)

        score = headroom * latency_score * (0.3 + 0.7 * success_rate) * concurrency_penalty
        if preferred_key_id and stats.key_id == preferred_key_id:
            score *= self.sticky_boost
        return score

    def available_count(self) -> int:
        now = time.monotonic()
        return sum(1 for k in self._keys if self._is_available(k, now))

    async def acquire(
        self,
        max_wait: float = 30.0,
        *,
        preferred_key_id: str | None = None,
    ) -> KeyStats:
        if not self._keys:
            raise RuntimeError("No NIM API keys configured. Set NIM_API_KEYS in the environment.")
        deadline = time.monotonic() + max_wait
        while True:
            async with self._lock:
                now = time.monotonic()
                available = [k for k in self._keys if self._is_available(k, now)]
                if available:
                    weights = [
                        max(0.01, self._score(k, now, preferred_key_id=preferred_key_id))
                        for k in available
                    ]
                    pick = random.choices(available, weights=weights, k=1)[0]
                    pick.in_flight += 1
                    pick.request_times.append(now)
                    pick.last_request_at = now
                    self._roll_daily(pick)
                    pick.daily_count += 1
                    logger.debug(
                        "acquired %s rpm=%s/%s in_flight=%s daily=%s score=%.3f",
                        pick.key_id,
                        self._rpm_used(pick, now),
                        self.rpm_limit,
                        pick.in_flight,
                        pick.daily_count,
                        self._score(pick, now, preferred_key_id=preferred_key_id),
                    )
                    return pick

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "All NIM API keys are rate-limited, quarantined, over budget, "
                    "or cooling down. Add more keys or wait."
                )
            # Sleep until the soonest key may become available (cap 2s) —
            # avoids 0.25s busy-spin under full cooldown.
            now2 = time.monotonic()
            waits = []
            for k in self._keys:
                if k.cooldown_until > now2:
                    waits.append(k.cooldown_until - now2)
                if k.quarantined_until > now2:
                    waits.append(k.quarantined_until - now2)
            sleep_for = min(waits) if waits else 0.25
            sleep_for = max(0.05, min(2.0, sleep_for))
            await asyncio.sleep(sleep_for)

    async def release(
        self,
        stats: KeyStats,
        *,
        success: bool,
        latency: float | None = None,
        rate_limited: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        async with self._lock:
            stats.in_flight = max(0, stats.in_flight - 1)

            if status_code in {401, 403}:
                stats.auth_failures += 1
                stats.error_count += 1
                if stats.auth_failures >= self.auth_fail_threshold:
                    stats.quarantined_until = time.monotonic() + self.auth_quarantine_seconds
                    logger.error(
                        "%s quarantined for %.0fs after %s auth failures (%s)",
                        stats.key_id,
                        self.auth_quarantine_seconds,
                        stats.auth_failures,
                        stats.mask(),
                    )
                return

            if rate_limited or status_code == 429:
                stats.rate_limit_hits += 1
                stats.error_count += 1
                cooldown = self.cooldown_seconds
                if retry_after_seconds is not None and retry_after_seconds > 0:
                    cooldown = max(cooldown, min(retry_after_seconds, 300.0))
                stats.cooldown_until = max(stats.cooldown_until, time.monotonic() + cooldown)
                logger.warning(
                    "%s hit rate limit; cooling down %.0fs",
                    stats.key_id,
                    cooldown,
                )
                return

            if success:
                stats.success_count += 1
                stats.auth_failures = 0  # clear latch so quarantine is time-boxed only
                if latency is not None:
                    stats.ewma_latency = 0.7 * stats.ewma_latency + 0.3 * latency
            else:
                stats.error_count += 1

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        out = []
        for k in self._keys:
            self._prune(k, now)
            self._roll_daily(k)
            out.append(
                {
                    "id": k.key_id,
                    "masked_key": k.mask(),
                    "rpm_used": len(k.request_times),
                    "rpm_limit": self.rpm_limit,
                    "in_flight": k.in_flight,
                    "ewma_latency_ms": round(k.ewma_latency * 1000, 1),
                    "success_count": k.success_count,
                    "error_count": k.error_count,
                    "rate_limit_hits": k.rate_limit_hits,
                    "cooling_down": now < k.cooldown_until,
                    "cooldown_remaining_s": max(0.0, round(k.cooldown_until - now, 1)),
                    "available": self._is_available(k, now),
                    "auth_failures": k.auth_failures,
                    "quarantined": now < k.quarantined_until,
                    "quarantine_remaining_s": max(0.0, round(k.quarantined_until - now, 1)),
                    "daily_count": k.daily_count,
                    "daily_limit": self.rpd_limit,
                    "daily_window": k.daily_window_start,
                }
            )
        return out
