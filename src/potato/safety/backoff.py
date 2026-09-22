"""Exponential backoff for 429 / transport / 5xx retries (not for model swaps)."""

from __future__ import annotations

import asyncio
import random


def compute_backoff_seconds(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 16.0,
    retry_after: float | None = None,
    max_delay: float | None = None,
) -> float:
    """
    attempt: 0-based retry index.
    Exponential delay capped at `cap`, but Retry-After always wins if larger
    (do not dump a rate-limited API).
    Adds up to 20% jitter so multi-key clients don't sync-thump.
    max_delay: hard upper bound so in-loop sleeps never overrun the request
    deadline (D5b). Applied after jitter.
    """
    exp = min(cap, base * (2 ** max(0, attempt)))
    delay = exp
    if retry_after is not None and retry_after > 0:
        delay = max(delay, float(retry_after))
    delay *= 1.0 + random.uniform(0.0, 0.2)
    if max_delay is not None:
        delay = min(delay, max(0.0, max_delay))
    return max(0.0, delay)


async def sleep_backoff(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 16.0,
    retry_after: float | None = None,
    max_delay: float | None = None,
) -> float:
    delay = compute_backoff_seconds(
        attempt, base=base, cap=cap, retry_after=retry_after, max_delay=max_delay
    )
    if delay > 0:
        await asyncio.sleep(delay)
    return delay
