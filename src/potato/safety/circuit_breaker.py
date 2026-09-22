"""Per-provider circuit breaker (half-open/open/closed states).

Open after 5 consecutive failures across all models for a provider.
Half-open: allow 1 request every 30s to probe.
Close on success.
"""

from __future__ import annotations

import logging
import time
from enum import StrEnum

logger = logging.getLogger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProviderCircuitBreaker:
    """Per-provider circuit breaker with half-open probing."""

    def __init__(
        self,
        failure_threshold: int = 6,
        recovery_timeout: float = 15.0,
        cooldown_multiplier: float = 2.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.cooldown_multiplier = cooldown_multiplier
        self._state: dict[str, BreakerState] = {}
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._last_probe: dict[str, float] = {}
        self._model_cooldowns: dict[str, float] = {}

    def is_model_on_cooldown(self, model_id: str) -> bool:
        """True if model is temporarily in cooldown."""
        until = self._model_cooldowns.get(model_id.lower(), 0.0)
        return time.monotonic() < until

    def model_fail(self, model_id: str, cooldown_seconds: float = 60.0) -> None:
        """Place an individual model on cooldown without penalizing the provider."""
        self._model_cooldowns[model_id.lower()] = time.monotonic() + cooldown_seconds
        logger.warning("model %s placed on cooldown for %.0fs", model_id, cooldown_seconds)

    def model_succeed(self, model_id: str) -> None:
        """Clear model cooldown on success."""
        self._model_cooldowns.pop(model_id.lower(), None)

    def allow(self, provider_id: str) -> bool:
        """True if a request may be sent to this provider."""
        pid = provider_id.lower()
        state = self._state.get(pid, BreakerState.CLOSED)
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.HALF_OPEN:
            # Only one in-flight probe at a time
            last = self._last_probe.get(pid, 0.0)
            if time.monotonic() - last < self.recovery_timeout:
                return False
            self._last_probe[pid] = time.monotonic()
            return True
        # OPEN: check if cooldown has elapsed → transition to half-open
        until = self._open_until.get(pid, 0)
        if time.monotonic() >= until:
            self._state[pid] = BreakerState.HALF_OPEN
            self._last_probe[pid] = time.monotonic()
            logger.info("circuit half-open → probing provider %s", pid)
            return True
        return False

    def blocked(self, provider_id: str) -> bool:
        """True when requests to this provider are hard-skipped (open + cooldown pending).

        Non-mutating: safe for chain building / availability checks, unlike
        ``allow()`` which consumes probe slots and transitions state.
        """
        pid = (provider_id or "").lower()
        if self._state.get(pid, BreakerState.CLOSED) != BreakerState.OPEN:
            return False
        return time.monotonic() < self._open_until.get(pid, 0.0)

    def any_closed(self, provider_ids: set[str] | list[str]) -> bool:
        """True if at least one provider is NOT hard-skipped (may serve now)."""
        return any(not self.blocked(pid) for pid in provider_ids)

    def force_allow(self, provider_id: str) -> None:
        """Last-resort: let a probe through immediately.

        Resets failure count so the next probe gets a clean backoff baseline
        (without this, accumulated failures cause exponentially growing backoff
        that makes recovery take hours after many consecutive failures).
        """
        pid = provider_id.lower()
        self._state[pid] = BreakerState.HALF_OPEN
        self._failures[pid] = 0
        self._open_until.pop(pid, None)

    def fail(
        self,
        provider_id: str,
        *,
        is_transport: bool = True,
        model_id: str | None = None,
    ) -> None:
        """Record a failure from this provider or model.
        
        Model-specific errors (404, 400, 429) place only that model on cooldown.
        Only transport/connectivity failures (502, 503, 504, connect timeout)
        penalize the provider circuit breaker.
        """
        if model_id and not is_transport:
            self.model_fail(model_id)
            return

        if not is_transport:
            return

        pid = provider_id.lower()
        self._failures[pid] = self._failures.get(pid, 0) + 1
        f = self._failures[pid]
        state = self._state.get(pid, BreakerState.CLOSED)

        if state == BreakerState.HALF_OPEN:
            # Probing failed → re-open with exponential backoff
            backoff = self.recovery_timeout * (
                self.cooldown_multiplier ** min(f - self.failure_threshold, 4)
            )
            self._state[pid] = BreakerState.OPEN
            self._open_until[pid] = time.monotonic() + backoff
            logger.warning(
                "circuit re-opened for provider %s (backoff=%.0fs, failures=%s)",
                pid,
                backoff,
                f,
            )
            return

        if state == BreakerState.CLOSED and f >= self.failure_threshold:
            self._state[pid] = BreakerState.OPEN
            self._open_until[pid] = time.monotonic() + self.recovery_timeout
            logger.warning(
                "circuit opened for provider %s (%s consecutive failures)",
                pid,
                f,
            )

    def succeed(self, provider_id: str, *, model_id: str | None = None) -> None:
        """Record a success → close the circuit."""
        if model_id:
            self.model_succeed(model_id)
        pid = provider_id.lower()
        self._state[pid] = BreakerState.CLOSED
        self._failures[pid] = 0
        self._open_until.pop(pid, None)
        if pid in self._last_probe:
            logger.info("circuit closed for provider %s (probe succeeded)", pid)

    def state(self, provider_id: str) -> BreakerState:
        return self._state.get(provider_id.lower(), BreakerState.CLOSED)

    def reset(self, provider_id: str) -> None:
        pid = provider_id.lower()
        self._state[pid] = BreakerState.CLOSED
        self._failures[pid] = 0
        self._open_until.pop(pid, None)

    def snapshot(self) -> dict[str, dict]:
        now = time.monotonic()
        out: dict[str, dict] = {}
        for pid in set(self._state.keys()) | set(self._failures.keys()):
            state = self._state.get(pid, BreakerState.CLOSED)
            out[pid] = {
                "state": state.value,
                "failures": self._failures.get(pid, 0),
                "open_until": max(0, round(self._open_until.get(pid, 0) - now, 1)),
            }
        return out
