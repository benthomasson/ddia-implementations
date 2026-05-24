"""Event Sourcing Store with Projections and Snapshots."""

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Optional


class ConcurrencyConflict(Exception):
    """Raised when optimistic concurrency check fails."""
    pass


@dataclass
class Event:
    """An immutable event in the event log."""
    event_id: int
    stream_id: str
    event_type: str
    data: dict
    timestamp: datetime
    metadata: Optional[dict] = None


class EventStore:
    """Append-only event store with optional disk persistence."""

    def __init__(self, persist_path: Optional[str] = None):
        self._events: list[Event] = []
        self._streams: dict[str, list[int]] = {}  # stream_id -> list of indices into _events
        self._persist_path = persist_path
        self._subscribers: list[Callable[[Event], None]] = []

        if persist_path and os.path.exists(persist_path):
            self._load_from_file(persist_path)

    def append(self, stream_id: str, event_type: str, data: dict,
               metadata: Optional[dict] = None,
               expected_version: Optional[int] = None) -> Event:
        """Append an event to a stream. Returns the stored event."""
        if expected_version is not None:
            current = self.stream_version(stream_id)
            if current != expected_version:
                raise ConcurrencyConflict(
                    f"Expected version {expected_version} but stream '{stream_id}' is at version {current}")

        event_id = len(self._events) + 1
        event = Event(
            event_id=event_id,
            stream_id=stream_id,
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self._events.append(event)
        self._streams.setdefault(stream_id, []).append(len(self._events) - 1)

        if self._persist_path:
            self._persist_event(event)

        for sub in self._subscribers:
            sub(event)

        return event

    def append_batch(self, stream_id: str, events: list[tuple[str, dict]],
                     expected_version: Optional[int] = None) -> list[Event]:
        """Atomically append multiple events to a stream."""
        if expected_version is not None:
            current = self.stream_version(stream_id)
            if current != expected_version:
                raise ConcurrencyConflict(
                    f"Expected version {expected_version} but stream '{stream_id}' is at version {current}")

        result = []
        for event_type, data in events:
            event_id = len(self._events) + 1
            event = Event(
                event_id=event_id,
                stream_id=stream_id,
                event_type=event_type,
                data=data,
                timestamp=datetime.now(),
            )
            self._events.append(event)
            self._streams.setdefault(stream_id, []).append(len(self._events) - 1)
            if self._persist_path:
                self._persist_event(event)
            result.append(event)

        for event in result:
            for sub in self._subscribers:
                sub(event)

        return result

    def read_stream(self, stream_id: str, from_version: int = 0) -> list[Event]:
        """Read events for a stream, optionally starting from a version."""
        indices = self._streams.get(stream_id, [])
        return [self._events[i] for i in indices if self._events[i].event_id >= from_version]

    def read_all(self, from_position: int = 0) -> list[Event]:
        """Read all events globally from a position."""
        return [e for e in self._events if e.event_id >= from_position]

    def stream_version(self, stream_id: str) -> int:
        """Return the current version (number of events) for a stream."""
        return len(self._streams.get(stream_id, []))

    def all_stream_ids(self) -> list[str]:
        """Return all known stream IDs."""
        return list(self._streams.keys())

    @property
    def global_position(self) -> int:
        """The current global position (total number of events)."""
        return len(self._events)

    def _persist_event(self, event: Event):
        with open(self._persist_path, "a") as f:
            record = {
                "event_id": event.event_id,
                "stream_id": event.stream_id,
                "event_type": event.event_type,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata,
            }
            f.write(json.dumps(record) + "\n")

    def _load_from_file(self, path: str):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                event = Event(
                    event_id=record["event_id"],
                    stream_id=record["stream_id"],
                    event_type=record["event_type"],
                    data=record["data"],
                    timestamp=datetime.fromisoformat(record["timestamp"]),
                    metadata=record.get("metadata"),
                )
                self._events.append(event)
                self._streams.setdefault(event.stream_id, []).append(len(self._events) - 1)


class Projection:
    """Processes events and maintains derived state."""

    def __init__(self, name: str, store: EventStore,
                 snapshot_interval: Optional[int] = None):
        self.name = name
        self._store = store
        self._handlers: dict[str, Callable] = {}
        self._state: dict = {}
        self._position: int = 0
        self._snapshot_interval = snapshot_interval
        self._events_since_snapshot = 0

    def when(self, event_type: str, handler: Callable[[dict, Event], None]):
        """Register a handler for an event type."""
        self._handlers[event_type] = handler

    def catch_up(self) -> int:
        """Process all unprocessed events. Returns number of events processed."""
        events = self._store.read_all(from_position=self._position + 1)
        count = 0
        for event in events:
            if event.event_type in self._handlers:
                self._handlers[event.event_type](self._state, event)
            self._position = event.event_id
            count += 1
            self._events_since_snapshot += 1
            if self._snapshot_interval and self._events_since_snapshot >= self._snapshot_interval:
                self.save_snapshot()
                self._events_since_snapshot = 0
        return count

    @property
    def state(self) -> dict:
        return self._state

    @property
    def position(self) -> int:
        return self._position

    def save_snapshot(self):
        """Save the current state as a snapshot."""
        if not hasattr(self._store, '_snapshots'):
            self._store._snapshots = {}
        self._store._snapshots[self.name] = {
            "state": copy.deepcopy(self._state),
            "position": self._position,
        }

    def load_snapshot(self) -> bool:
        """Load the latest snapshot. Returns True if found."""
        snapshots = getattr(self._store, '_snapshots', {})
        if self.name not in snapshots:
            return False
        snap = snapshots[self.name]
        self._state = copy.deepcopy(snap["state"])
        self._position = snap["position"]
        return True

    def reset(self):
        """Reset the projection state and position."""
        self._state = {}
        self._position = 0
        self._events_since_snapshot = 0


class LiveProjection(Projection):
    """A projection that automatically updates when events are appended."""

    def __init__(self, name: str, store: EventStore,
                 snapshot_interval: Optional[int] = None):
        super().__init__(name, store, snapshot_interval)
        store._subscribers.append(self._on_event)

    def _on_event(self, event: Event):
        if event.event_id <= self._position:
            return
        if event.event_type in self._handlers:
            self._handlers[event.event_type](self._state, event)
        self._position = event.event_id
        self._events_since_snapshot += 1
        if self._snapshot_interval and self._events_since_snapshot >= self._snapshot_interval:
            self.save_snapshot()
            self._events_since_snapshot = 0


def reconstruct_state(store: EventStore, stream_id: str,
                      handlers: dict[str, Callable],
                      up_to: Optional[int] = None) -> dict:
    """Reconstruct state of a stream by replaying events up to a given event_id."""
    events = store.read_stream(stream_id)
    state = {}
    for event in events:
        if up_to is not None and event.event_id > up_to:
            break
        if event.event_type in handlers:
            handlers[event.event_type](state, event)
    return state
