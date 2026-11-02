"""Core sliding-window rate limiter with a priority queue.

Accounting model:
- `_initiations` holds the monotonic timestamp of every request that has
  actually fired. A new request may fire only when fewer than `max_rpm`
  timestamps fall inside the trailing `window_size` seconds. This is a true
  sliding window, so traffic cannot burst across a window boundary.
- A slot is consumed the moment a request fires and is NEVER returned when the
  request completes, because NVIDIA already counted the initiation.
- Requests still streaming live in `in_flight` for abort/timeout tracking only.
- The priority queue is a min-heap: lower number = higher priority. Ties break
  by arrival timestamp, then by a monotonic sequence number (FIFO).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple


class QueueFull(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"queue full, retry after {retry_after}s")
        self.retry_after = retry_after


@dataclass
class Slot:
    request_id: str
    initiated_at: float
    priority: int
    entry: "Entry" = field(repr=False)


@dataclass
class Entry:
    priority: int
    arrival_ts: float
    seq: int
    fired: asyncio.Event = field(default_factory=asyncio.Event)
    request_id: str = ""
    initiated_at: float = 0.0
    slot: Optional[Slot] = None
    cancelled: bool = False


class RateLimiter:
    def __init__(
        self,
        max_rpm: int = 40,
        window_size: float = 60.0,
        max_queue: int = 500,
        retry_ttl: float = 300.0,
        sweep_interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_rpm = max_rpm
        self.window_size = window_size
        self.max_queue = max_queue
        self.retry_ttl = retry_ttl
        self.sweep_interval = sweep_interval
        self._clock = clock
        self._seq = itertools.count()
        self._initiations: Deque[float] = deque()
        self.queue: List[Tuple[int, float, int, Entry]] = []
        self.in_flight: Dict[str, Entry] = {}
        self._lock = asyncio.Lock()
        self.total_fired = 0
        self.total_rejected = 0
        self.total_cancelled = 0

    def _prune(self, now: float) -> None:
        """Drop initiations that have aged out of the trailing window."""
        cutoff = now - self.window_size
        while self._initiations and self._initiations[0] <= cutoff:
            self._initiations.popleft()

    @property
    def window_count(self) -> int:
        return len(self._initiations)

    def next_slot_in(self, now: Optional[float] = None) -> float:
        """Seconds until one more request may fire.

        Returns 0.0 when the current window still has room.
        """
        now = now if now is not None else self._clock()
        self._prune(now)
        if not self._initiations or len(self._initiations) < self.max_rpm:
            return 0.0
        return max(0.0, self._initiations[0] + self.window_size - now)

    def window_resets_in(self, now: Optional[float] = None) -> float:
        """Seconds until the oldest initiation ages out of the window."""
        now = now if now is not None else self._clock()
        self._prune(now)
        if not self._initiations:
            return 0.0
        return max(0.0, self._initiations[0] + self.window_size - now)

    def _grant(self, entry: Entry, now: float) -> Slot:
        entry.request_id = uuid.uuid4().hex
        entry.initiated_at = now
        entry.slot = Slot(
            request_id=entry.request_id,
            initiated_at=now,
            priority=entry.priority,
            entry=entry,
        )
        self._initiations.append(now)
        self.total_fired += 1
        self.in_flight[entry.request_id] = entry
        entry.fired.set()
        return entry.slot

    def _remove_from_queue(self, entry: Entry) -> None:
        if not self.queue:
            return
        self.queue = [it for it in self.queue if it[3] is not entry]
        heapq.heapify(self.queue)

    def _promote_starved(self, now: float) -> None:
        changed = False
        for item in self.queue:
            entry = item[3]
            if now - entry.arrival_ts >= self.retry_ttl and entry.priority > 1:
                entry.priority = 1
                changed = True
        if changed:
            self.queue = [(e.priority, e.arrival_ts, e.seq, e) for (_, _, _, e) in self.queue]
            heapq.heapify(self.queue)

    async def _drain_locked(self) -> None:
        now = self._clock()
        self._prune(now)
        self._promote_starved(now)
        available = self.max_rpm - len(self._initiations)
        while available > 0 and self.queue:
            _, _, _, entry = heapq.heappop(self.queue)
            if entry.cancelled:
                self.total_cancelled += 1
                continue
            self._grant(entry, now)
            available -= 1

    async def drain(self) -> None:
        async with self._lock:
            await self._drain_locked()

    async def acquire(self, priority: int = 3) -> Slot:
        entry = Entry(
            priority=priority,
            arrival_ts=self._clock(),
            seq=next(self._seq),
        )
        while True:
            now = self._clock()
            async with self._lock:
                self._prune(now)
                if self.queue:
                    await self._drain_locked()
                    now = self._clock()
                if not self.queue and len(self._initiations) < self.max_rpm:
                    return self._grant(entry, now)
                if len(self.queue) >= self.max_queue:
                    self.total_rejected += 1
                    retry_after = max(1, int(self.next_slot_in(now)))
                    raise QueueFull(retry_after)
                heapq.heappush(self.queue, (entry.priority, entry.arrival_ts, entry.seq, entry))
            try:
                await entry.fired.wait()
            except asyncio.CancelledError:
                async with self._lock:
                    if entry.slot is None:
                        entry.cancelled = True
                        self._remove_from_queue(entry)
                        self.total_cancelled += 1
                raise
            async with self._lock:
                if entry.slot is not None:
                    return entry.slot

    async def release(self, slot: Slot) -> None:
        async with self._lock:
            self.in_flight.pop(slot.request_id, None)
            await self._drain_locked()

    async def _force_release_stale(self, now: float, max_age: float) -> int:
        stale = [e for e in self.in_flight.values() if now - e.initiated_at > max_age]
        for entry in stale:
            self.in_flight.pop(entry.request_id, None)
            self.total_cancelled += 1
        return len(stale)

    async def sweep(self, max_stream_age: float) -> None:
        async with self._lock:
            now = self._clock()
            await self._drain_locked()
            await self._force_release_stale(now, max_stream_age)

    async def _sweep_loop(self, max_stream_age: float) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval)
            await self.sweep(max_stream_age)

    def start(self, max_stream_age: float) -> asyncio.Task:
        return asyncio.create_task(self._sweep_loop(max_stream_age))

    @property
    def queue_depth(self) -> int:
        return len(self.queue)

    def status(self) -> dict:
        now = self._clock()
        self._prune(now)
        return {
            "max_rpm": self.max_rpm,
            "window_size": self.window_size,
            "window_count": len(self._initiations),
            "window_resets_in": round(self.next_slot_in(now), 2),
            "queue_depth": len(self.queue),
            "in_flight": len(self.in_flight),
            "total_fired": self.total_fired,
            "total_rejected": self.total_rejected,
            "total_cancelled": self.total_cancelled,
        }
