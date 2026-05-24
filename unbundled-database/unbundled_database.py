"""Composable unbundled database implementing the 'database inside-out' pattern from DDIA Ch.12."""

from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class WALEntry:
    lsn: int
    operation: str  # "PUT" or "DELETE"
    key: str
    value: Optional[dict]  # None for DELETE


@dataclass
class CDCEvent:
    lsn: int
    operation: str  # "insert", "update", or "delete"
    key: str
    new_value: Optional[dict]
    old_value: Optional[dict]


class WriteAheadLog:
    """Append-only log of all mutation operations."""

    def __init__(self, persist_path: Optional[str] = None):
        self._entries: list[WALEntry] = []
        self._next_lsn = 1
        self._persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            with open(persist_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    entry = WALEntry(**data)
                    self._entries.append(entry)
                    self._next_lsn = entry.lsn + 1

    def append(self, operation: str, key: str, value: Optional[dict] = None) -> WALEntry:
        """Append an entry to the log. Returns the entry with its LSN."""
        entry = WALEntry(lsn=self._next_lsn, operation=operation, key=key, value=value)
        self._next_lsn += 1
        self._entries.append(entry)
        if self._persist_path:
            with open(self._persist_path, "a") as f:
                f.write(json.dumps({"lsn": entry.lsn, "operation": entry.operation,
                                    "key": entry.key, "value": entry.value}) + "\n")
        return entry

    def read_from(self, lsn: int) -> list[WALEntry]:
        """Read all entries from the given LSN onward."""
        return [e for e in self._entries if e.lsn >= lsn]

    def truncate_before(self, lsn: int) -> int:
        """Discard entries before the given LSN. Returns count removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.lsn >= lsn]
        return before - len(self._entries)

    @property
    def latest_lsn(self) -> int:
        return self._entries[-1].lsn if self._entries else 0

    @property
    def earliest_lsn(self) -> int:
        return self._entries[0].lsn if self._entries else 0

    def __len__(self) -> int:
        return len(self._entries)


class StorageEngine:
    """Key-value store that only mutates via WAL replay."""

    def __init__(self):
        self._data: dict[str, dict] = {}
        self._current_lsn = 0

    def apply(self, entry: WALEntry) -> None:
        """Apply a WAL entry to update the store."""
        if entry.operation == "PUT":
            self._data[entry.key] = entry.value
        elif entry.operation == "DELETE":
            self._data.pop(entry.key, None)
        self._current_lsn = entry.lsn

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def scan(self, prefix: str = "") -> dict[str, dict]:
        return {k: v for k, v in self._data.items() if k.startswith(prefix)}

    @property
    def current_lsn(self) -> int:
        return self._current_lsn

    @property
    def record_count(self) -> int:
        return len(self._data)

    def rebuild(self, wal: WriteAheadLog) -> None:
        """Clear state and rebuild by replaying the entire WAL."""
        self._data.clear()
        self._current_lsn = 0
        for entry in wal.read_from(wal.earliest_lsn):
            self.apply(entry)


class DerivedSystem(ABC):
    """Base class for derived data systems."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def position(self) -> int:
        ...

    @abstractmethod
    def process_event(self, event: CDCEvent) -> None:
        ...

    @abstractmethod
    def rebuild(self, events: list[CDCEvent]) -> None:
        ...

    @abstractmethod
    def get_state(self) -> Any:
        ...


class SecondaryIndex(DerivedSystem):
    """Inverted index mapping field values to sets of primary keys."""

    def __init__(self, name: str, indexed_fields: list[str]):
        self._name = name
        self._indexed_fields = indexed_fields
        self._index: dict[str, dict[Any, set[str]]] = {f: {} for f in indexed_fields}
        self._position = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> int:
        return self._position

    def process_event(self, event: CDCEvent) -> None:
        if event.operation in ("update", "delete"):
            self._remove(event.key, event.old_value)
        if event.operation in ("insert", "update"):
            self._add(event.key, event.new_value)
        self._position = event.lsn

    def _add(self, key: str, value: Optional[dict]) -> None:
        if not value:
            return
        for f in self._indexed_fields:
            if f in value:
                v = value[f]
                self._index[f].setdefault(v, set()).add(key)

    def _remove(self, key: str, value: Optional[dict]) -> None:
        if not value:
            return
        for f in self._indexed_fields:
            if f in value:
                v = value[f]
                if v in self._index[f]:
                    self._index[f][v].discard(key)
                    if not self._index[f][v]:
                        del self._index[f][v]

    def rebuild(self, events: list[CDCEvent]) -> None:
        self._index = {f: {} for f in self._indexed_fields}
        self._position = 0
        for event in events:
            self.process_event(event)

    def query(self, field: str, value: Any) -> list[str]:
        return sorted(self._index.get(field, {}).get(value, set()))

    def get_state(self) -> Any:
        return {f: {v: sorted(keys) for v, keys in vals.items()} for f, vals in self._index.items()}


class MaterializedView(DerivedSystem):
    """Pre-computed aggregate grouped by a field."""

    def __init__(self, name: str, group_by_field: str, aggregate: str = "count"):
        self._name = name
        self._group_by_field = group_by_field
        self._aggregate = aggregate
        self._counts: dict[Any, int] = {}
        self._lists: dict[Any, list[str]] = {}
        self._position = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> int:
        return self._position

    def process_event(self, event: CDCEvent) -> None:
        if event.operation in ("update", "delete") and event.old_value:
            old_group = event.old_value.get(self._group_by_field)
            if old_group is not None:
                if self._aggregate == "count":
                    self._counts[old_group] = max(0, self._counts.get(old_group, 0) - 1)
                elif self._aggregate == "list":
                    if old_group in self._lists:
                        try:
                            self._lists[old_group].remove(event.key)
                        except ValueError:
                            pass

        if event.operation in ("insert", "update") and event.new_value:
            new_group = event.new_value.get(self._group_by_field)
            if new_group is not None:
                if self._aggregate == "count":
                    self._counts[new_group] = self._counts.get(new_group, 0) + 1
                elif self._aggregate == "list":
                    self._lists.setdefault(new_group, []).append(event.key)

        self._position = event.lsn

    def rebuild(self, events: list[CDCEvent]) -> None:
        self._counts.clear()
        self._lists.clear()
        self._position = 0
        for event in events:
            self.process_event(event)

    def query(self, value: Any) -> Any:
        if self._aggregate == "count":
            return self._counts.get(value, 0)
        elif self._aggregate == "list":
            return list(self._lists.get(value, []))

    def get_state(self) -> Any:
        if self._aggregate == "count":
            return dict(self._counts)
        return {k: list(v) for k, v in self._lists.items()}


class FullTextSearch(DerivedSystem):
    """Simple inverted index on text field content."""

    def __init__(self, name: str, text_fields: list[str]):
        self._name = name
        self._text_fields = text_fields
        self._index: dict[str, set[str]] = {}  # word -> set of keys
        self._position = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> int:
        return self._position

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _extract_words(self, value: dict) -> set[str]:
        words = set()
        for f in self._text_fields:
            if f in value and isinstance(value[f], str):
                words.update(self._tokenize(value[f]))
        return words

    def process_event(self, event: CDCEvent) -> None:
        if event.operation in ("update", "delete") and event.old_value:
            old_words = self._extract_words(event.old_value)
            for w in old_words:
                if w in self._index:
                    self._index[w].discard(event.key)
                    if not self._index[w]:
                        del self._index[w]

        if event.operation in ("insert", "update") and event.new_value:
            new_words = self._extract_words(event.new_value)
            for w in new_words:
                self._index.setdefault(w, set()).add(event.key)

        self._position = event.lsn

    def rebuild(self, events: list[CDCEvent]) -> None:
        self._index.clear()
        self._position = 0
        for event in events:
            self.process_event(event)

    def search(self, term: str) -> list[str]:
        return sorted(self._index.get(term.lower(), set()))

    def search_all(self, terms: list[str]) -> list[str]:
        if not terms:
            return []
        sets = [self._index.get(t.lower(), set()) for t in terms]
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return sorted(result)

    def get_state(self) -> Any:
        return {w: sorted(keys) for w, keys in self._index.items()}


class CDCStream:
    """Change data capture stream derived from the WAL."""

    def __init__(self, wal: WriteAheadLog, storage: StorageEngine):
        self._wal = wal
        self._storage = storage
        self._events: list[CDCEvent] = []
        self._consumers: dict[str, DerivedSystem] = {}

    def emit(self, entry: WALEntry, old_value: Optional[dict] = None) -> CDCEvent:
        """Convert a WAL entry into a CDC event."""
        if entry.operation == "PUT":
            op = "update" if old_value is not None else "insert"
        else:
            op = "delete"

        event = CDCEvent(
            lsn=entry.lsn,
            operation=op,
            key=entry.key,
            new_value=entry.value,
            old_value=old_value,
        )
        self._events.append(event)
        return event

    def subscribe(self, consumer: DerivedSystem) -> None:
        self._consumers[consumer.name] = consumer

    def unsubscribe(self, consumer_name: str) -> None:
        self._consumers.pop(consumer_name, None)

    def process_pending(self) -> dict[str, int]:
        """Process all pending events for all consumers."""
        result = {}
        for name, consumer in self._consumers.items():
            count = 0
            for event in self._events:
                if event.lsn > consumer.position:
                    consumer.process_event(event)
                    count += 1
            result[name] = count
        return result

    def get_lag(self) -> dict[str, int]:
        latest = self._events[-1].lsn if self._events else 0
        return {name: latest - consumer.position for name, consumer in self._consumers.items()}

    def snapshot_and_stream(self, consumer: DerivedSystem) -> int:
        """Send a full snapshot of current state, then subscribe for live changes."""
        snapshot_count = 0
        # Synthesize insert events from current storage state
        for key, value in sorted(self._storage.scan("").items()):
            snapshot_event = CDCEvent(
                lsn=0,
                operation="insert",
                key=key,
                new_value=value,
                old_value=None,
            )
            consumer.process_event(snapshot_event)
            snapshot_count += 1

        # Set position to latest so it only gets future events
        if self._events:
            # Process any events that happened after the storage state
            # Set position to latest CDC event LSN
            consumer._position = self._events[-1].lsn
        self.subscribe(consumer)
        return snapshot_count

    @property
    def events(self) -> list[CDCEvent]:
        return self._events


class UnbundledDatabase:
    """Facade wiring together WAL, StorageEngine, CDCStream, and derived systems."""

    def __init__(self, persist_dir: Optional[str] = None):
        wal_path = None
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            wal_path = os.path.join(persist_dir, "wal.jsonl")
        self._wal = WriteAheadLog(persist_path=wal_path)
        self._storage = StorageEngine()
        self._cdc = CDCStream(self._wal, self._storage)
        self._derived_systems: dict[str, DerivedSystem] = {}

    def put(self, key: str, value: dict) -> CDCEvent:
        """Insert or update a record."""
        old_value = self._storage.get(key)
        entry = self._wal.append("PUT", key, value)
        self._storage.apply(entry)
        event = self._cdc.emit(entry, old_value)
        return event

    def get(self, key: str) -> Optional[dict]:
        return self._storage.get(key)

    def delete(self, key: str) -> Optional[CDCEvent]:
        """Delete a record by primary key."""
        old_value = self._storage.get(key)
        if old_value is None:
            return None
        entry = self._wal.append("DELETE", key)
        self._storage.apply(entry)
        event = self._cdc.emit(entry, old_value)
        return event

    def add_derived_system(self, system: DerivedSystem, catch_up: bool = True) -> None:
        """Add a derived data system, optionally catching up on existing data."""
        self._derived_systems[system.name] = system
        if catch_up and self._cdc.events:
            self._cdc.snapshot_and_stream(system)
        else:
            self._cdc.subscribe(system)

    def query_index(self, index_name: str, field: str, value: Any) -> list[dict]:
        """Query a secondary index and return full records."""
        system = self._derived_systems[index_name]
        keys = system.query(field, value)
        return [self._storage.get(k) for k in keys if self._storage.get(k) is not None]

    def flush(self) -> dict[str, int]:
        return self._cdc.process_pending()

    def get_lag(self) -> dict[str, int]:
        return self._cdc.get_lag()

    def rebuild_system(self, system_name: str) -> int:
        """Rebuild a derived system from scratch by replaying CDC."""
        system = self._derived_systems[system_name]
        events = self._cdc.events
        system.rebuild(events)
        return len(events)

    def get_pipeline_state(self) -> dict:
        lag = self.get_lag()
        return {
            "wal_size": len(self._wal),
            "storage_records": self._storage.record_count,
            "cdc_events": len(self._cdc.events),
            "derived_systems": [
                {"name": name, "position": sys.position, "lag": lag.get(name, 0)}
                for name, sys in self._derived_systems.items()
            ],
        }

    @property
    def wal(self) -> WriteAheadLog:
        return self._wal

    @property
    def storage(self) -> StorageEngine:
        return self._storage

    @property
    def cdc(self) -> CDCStream:
        return self._cdc
