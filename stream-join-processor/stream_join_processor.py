"""Time-windowed stream join processor supporting inner, left, and full outer joins."""

from typing import Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict


class JoinType(Enum):
    INNER = "inner"
    LEFT = "left"
    FULL_OUTER = "full_outer"


@dataclass
class StreamEvent:
    """An event in a stream with a join key and timestamp."""
    stream_name: str
    key: str
    value: dict
    timestamp: float


@dataclass
class JoinResult:
    """Result of joining two events (or a miss with None on one side)."""
    key: str
    left_event: Optional[StreamEvent]
    right_event: Optional[StreamEvent]
    join_timestamp: float


@dataclass
class JoinStats:
    """Processing statistics."""
    left_events_processed: int = 0
    right_events_processed: int = 0
    matches_emitted: int = 0
    misses_emitted: int = 0
    late_events_dropped: int = 0


class TimeWindow:
    """A time window of fixed duration."""

    def __init__(self, duration_seconds: float):
        self.duration = duration_seconds

    def contains(self, t1: float, t2: float) -> bool:
        """Check if two timestamps fall within this window."""
        return abs(t1 - t2) <= self.duration


@dataclass
class _BufferedEvent:
    """Internal wrapper tracking match status."""
    event: StreamEvent
    uid: int
    matched: bool = False


class StreamJoinProcessor:
    """Joins two event streams on a shared key within a time window."""

    def __init__(self, left_stream: str, right_stream: str,
                 window: TimeWindow,
                 join_type: JoinType = JoinType.INNER,
                 allowed_lateness: float = 0.0,
                 on_result: Optional[Callable[[JoinResult], None]] = None):
        self._left_stream = left_stream
        self._right_stream = right_stream
        self._window = window
        self._join_type = join_type
        self._allowed_lateness = allowed_lateness
        self._on_result = on_result

        # Buffers: key -> list of _BufferedEvent
        self._left_buffer: dict[str, list[_BufferedEvent]] = defaultdict(list)
        self._right_buffer: dict[str, list[_BufferedEvent]] = defaultdict(list)

        self._watermark = -float('inf')
        self._stats = JoinStats()
        self._results: list[JoinResult] = []
        self._uid_counter = 0

    def _next_uid(self) -> int:
        self._uid_counter += 1
        return self._uid_counter

    def _emit(self, result: JoinResult):
        self._results.append(result)
        if self._on_result:
            self._on_result(result)

    def _is_left(self, event: StreamEvent) -> bool:
        return event.stream_name == self._left_stream

    def process_event(self, event: StreamEvent) -> list[JoinResult]:
        """Process a single event. Returns join results produced immediately."""
        is_left = self._is_left(event)

        # Update stats
        if is_left:
            self._stats.left_events_processed += 1
        else:
            self._stats.right_events_processed += 1

        # Check for late events
        if event.timestamp < self._watermark - self._allowed_lateness:
            self._stats.late_events_dropped += 1
            return []

        # Advance watermark
        new_watermark = max(self._watermark, event.timestamp)

        # Buffer the event
        buffered = _BufferedEvent(event=event, uid=self._next_uid())
        if is_left:
            self._left_buffer[event.key].append(buffered)
            other_buffer = self._right_buffer
        else:
            self._right_buffer[event.key].append(buffered)
            other_buffer = self._left_buffer

        # Check for matches in opposite buffer
        results_before = len(self._results)
        if event.key in other_buffer:
            for other in other_buffer[event.key]:
                if self._window.contains(event.timestamp, other.event.timestamp):
                    if is_left:
                        result = JoinResult(
                            key=event.key,
                            left_event=event,
                            right_event=other.event,
                            join_timestamp=max(event.timestamp, other.event.timestamp),
                        )
                    else:
                        result = JoinResult(
                            key=event.key,
                            left_event=other.event,
                            right_event=event,
                            join_timestamp=max(event.timestamp, other.event.timestamp),
                        )
                    buffered.matched = True
                    other.matched = True
                    self._stats.matches_emitted += 1
                    self._emit(result)

        # Now advance watermark and expire
        self._watermark = new_watermark
        self._expire_events()

        return self._results[results_before:]

    def advance_time(self, timestamp: float) -> list[JoinResult]:
        """Advance watermark and trigger expiration. Returns miss results."""
        if timestamp <= self._watermark:
            return []
        self._watermark = timestamp
        results_before = len(self._results)
        self._expire_events()
        return self._results[results_before:]

    def _expire_events(self):
        """Remove events older than the window and emit misses for outer joins."""
        cutoff = self._watermark - self._window.duration

        for buf, is_left_buf in [(self._left_buffer, True), (self._right_buffer, False)]:
            keys_to_clean = list(buf.keys())
            for key in keys_to_clean:
                events = buf[key]
                remaining = []
                for be in events:
                    if be.event.timestamp < cutoff:
                        # Expiring this event
                        if not be.matched:
                            should_emit_miss = False
                            if self._join_type == JoinType.LEFT and is_left_buf:
                                should_emit_miss = True
                            elif self._join_type == JoinType.FULL_OUTER:
                                should_emit_miss = True

                            if should_emit_miss:
                                if is_left_buf:
                                    result = JoinResult(
                                        key=key,
                                        left_event=be.event,
                                        right_event=None,
                                        join_timestamp=be.event.timestamp,
                                    )
                                else:
                                    result = JoinResult(
                                        key=key,
                                        left_event=None,
                                        right_event=be.event,
                                        join_timestamp=be.event.timestamp,
                                    )
                                self._stats.misses_emitted += 1
                                self._emit(result)
                    else:
                        remaining.append(be)
                if remaining:
                    buf[key] = remaining
                else:
                    del buf[key]

    def get_results(self) -> list[JoinResult]:
        """Retrieve and clear all buffered results."""
        results = self._results
        self._results = []
        return results

    @property
    def stats(self) -> JoinStats:
        return self._stats

    @property
    def buffer_size(self) -> tuple[int, int]:
        left = sum(len(v) for v in self._left_buffer.values())
        right = sum(len(v) for v in self._right_buffer.values())
        return (left, right)


class TumblingWindowAggregator:
    """Aggregates join results into tumbling (non-overlapping) time windows."""

    def __init__(self, window_seconds: float,
                 aggregate_fn: Callable[[str, list[JoinResult]], Any]):
        self._window_seconds = window_seconds
        self._aggregate_fn = aggregate_fn
        # (key, window_start) -> list of results
        self._windows: dict[tuple[str, float], list[JoinResult]] = defaultdict(list)

    def _window_start(self, timestamp: float) -> float:
        return (timestamp // self._window_seconds) * self._window_seconds

    def add(self, result: JoinResult):
        """Add a join result to the appropriate window."""
        ws = self._window_start(result.join_timestamp)
        self._windows[(result.key, ws)].append(result)

    def advance_time(self, timestamp: float) -> list[tuple[str, float, float, Any]]:
        """Close completed windows and return aggregated results."""
        current_ws = self._window_start(timestamp)
        completed = []
        keys_to_remove = []

        for (key, ws), results in self._windows.items():
            if ws + self._window_seconds <= current_ws:
                agg = self._aggregate_fn(key, results)
                completed.append((key, ws, ws + self._window_seconds, agg))
                keys_to_remove.append((key, ws))

        for k in keys_to_remove:
            del self._windows[k]

        completed.sort(key=lambda x: (x[1], x[0]))
        return completed
