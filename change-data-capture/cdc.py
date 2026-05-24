"""Change Data Capture (CDC) system — in-memory database with append-only change log."""

from typing import Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class Operation(Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class ChangeEvent:
    """A single change captured from the database."""
    sequence_number: int
    table: str
    operation: Operation
    key: Any
    before: Optional[dict]
    after: Optional[dict]
    timestamp: datetime


class CDCLog:
    """Ordered, append-only log of ChangeEvents."""

    def __init__(self):
        self._events: list[ChangeEvent] = []
        self._next_seq = 0

    def _append(self, table: str, operation: Operation, key: Any,
                before: Optional[dict], after: Optional[dict]) -> ChangeEvent:
        event = ChangeEvent(
            sequence_number=self._next_seq,
            table=table,
            operation=operation,
            key=key,
            before=before,
            after=after,
            timestamp=datetime.now(),
        )
        self._events.append(event)
        self._next_seq += 1
        return event

    def read_from(self, position: int = 0) -> list[ChangeEvent]:
        """Read all events from the given position onward."""
        return [e for e in self._events if e.sequence_number >= position]

    def read_range(self, start: int, end: int) -> list[ChangeEvent]:
        """Read events in [start, end) range."""
        return [e for e in self._events if start <= e.sequence_number < end]

    @property
    def current_position(self) -> int:
        """The sequence number of the latest event, or -1 if empty."""
        if not self._events:
            return -1
        return self._events[-1].sequence_number

    def compact(self) -> int:
        """Keep only the latest event per (table, key). Returns number removed."""
        seen: dict[tuple[str, Any], int] = {}
        # Walk backwards to find latest event index for each (table, key)
        for i in range(len(self._events) - 1, -1, -1):
            e = self._events[i]
            k = (e.table, e.key)
            if k not in seen:
                seen[k] = i
        keep = sorted(seen.values())
        removed = len(self._events) - len(keep)
        self._events = [self._events[i] for i in keep]
        return removed


class CDCDatabase:
    """Simple in-memory relational database with CDC."""

    def __init__(self):
        self._tables: dict[str, dict[str, Any]] = {}  # name -> {columns, pk, rows}
        self._log = CDCLog()

    def create_table(self, name: str, columns: list[str], primary_key: str):
        """Define a new table."""
        self._tables[name] = {
            "columns": columns,
            "pk": primary_key,
            "rows": {},
        }

    def _get_table(self, name: str) -> dict:
        if name not in self._tables:
            raise KeyError(f"Table '{name}' does not exist")
        return self._tables[name]

    def insert(self, table: str, row: dict):
        """Insert a row."""
        t = self._get_table(table)
        pk = row[t["pk"]]
        if pk in t["rows"]:
            raise ValueError(f"Duplicate primary key: {pk}")
        t["rows"][pk] = dict(row)
        self._log._append(table, Operation.INSERT, pk, None, dict(row))

    def update(self, table: str, pk_value: Any, changes: dict):
        """Update specific columns of a row."""
        t = self._get_table(table)
        if pk_value not in t["rows"]:
            raise KeyError(f"Row with key {pk_value} not found in '{table}'")
        before = dict(t["rows"][pk_value])
        t["rows"][pk_value].update(changes)
        after = dict(t["rows"][pk_value])
        self._log._append(table, Operation.UPDATE, pk_value, before, after)

    def delete(self, table: str, pk_value: Any):
        """Delete a row by primary key."""
        t = self._get_table(table)
        if pk_value not in t["rows"]:
            raise KeyError(f"Row with key {pk_value} not found in '{table}'")
        before = dict(t["rows"].pop(pk_value))
        self._log._append(table, Operation.DELETE, pk_value, before, None)

    def select(self, table: str, pk_value: Any) -> Optional[dict]:
        """Read a single row by primary key."""
        t = self._get_table(table)
        row = t["rows"].get(pk_value)
        return dict(row) if row is not None else None

    def scan(self, table: str) -> list[dict]:
        """Read all rows in a table."""
        t = self._get_table(table)
        return [dict(r) for r in t["rows"].values()]

    @property
    def cdc_log(self) -> CDCLog:
        return self._log


class CDCConsumer:
    """Subscribes to CDC log and processes events."""

    def __init__(self, name: str, log: CDCLog):
        self.name = name
        self._log = log
        self._position = 0
        self._handlers: list[tuple[Optional[str], Optional[Operation], Callable]] = []

    def on(self, table: str, operation: Optional[Operation],
           handler: Callable[[ChangeEvent], None]):
        """Register handler for events on a specific table/operation."""
        self._handlers.append((table, operation, handler))

    def on_all(self, handler: Callable[[ChangeEvent], None]):
        """Register handler for all events."""
        self._handlers.append((None, None, handler))

    def poll(self) -> int:
        """Fetch and process new events. Returns count processed."""
        events = self._log.read_from(self._position)
        for event in events:
            for tbl, op, handler in self._handlers:
                if tbl is not None and tbl != event.table:
                    continue
                if op is not None and op != event.operation:
                    continue
                handler(event)
        self._position = self._position + len(events)
        return len(events)

    @property
    def position(self) -> int:
        return self._position

    def seek(self, position: int):
        """Reset consumer position."""
        self._position = position


class MaterializedView:
    """Maintains a live copy of a table from the CDC stream."""

    def __init__(self, name: str, source_table: str, log: CDCLog,
                 transform: Optional[Callable[[dict], Optional[dict]]] = None):
        self.name = name
        self._source_table = source_table
        self._log = log
        self._transform = transform
        self._data: dict[Any, dict] = {}
        self._position = 0

    def refresh(self) -> int:
        """Process new CDC events. Returns count processed."""
        events = self._log.read_from(self._position)
        count = 0
        for event in events:
            if event.table == self._source_table:
                if event.operation == Operation.INSERT or event.operation == Operation.UPDATE:
                    row = dict(event.after)
                    if self._transform:
                        row = self._transform(row)
                    if row is not None:
                        self._data[event.key] = row
                    else:
                        self._data.pop(event.key, None)
                elif event.operation == Operation.DELETE:
                    self._data.pop(event.key, None)
            count += 1
        self._position += count
        return count

    def get(self, pk_value: Any) -> Optional[dict]:
        row = self._data.get(pk_value)
        return dict(row) if row is not None else None

    def scan(self) -> list[dict]:
        return [dict(r) for r in self._data.values()]


class SearchIndex:
    """Keyword-searchable index on specified columns, updated from CDC."""

    def __init__(self, name: str, source_table: str, log: CDCLog,
                 indexed_columns: list[str]):
        self.name = name
        self._source_table = source_table
        self._log = log
        self._indexed_columns = indexed_columns
        self._rows: dict[Any, dict] = {}
        self._index: dict[str, set[Any]] = {}  # token -> set of PKs
        self._position = 0

    def _tokenize(self, row: dict) -> set[str]:
        tokens = set()
        for col in self._indexed_columns:
            val = row.get(col, "")
            if val is not None:
                tokens.update(str(val).lower().split())
        return tokens

    def _remove_from_index(self, key: Any):
        old_row = self._rows.get(key)
        if old_row:
            for token in self._tokenize(old_row):
                if token in self._index:
                    self._index[token].discard(key)
                    if not self._index[token]:
                        del self._index[token]
            del self._rows[key]

    def _add_to_index(self, key: Any, row: dict):
        self._rows[key] = dict(row)
        for token in self._tokenize(row):
            if token not in self._index:
                self._index[token] = set()
            self._index[token].add(key)

    def refresh(self) -> int:
        """Process new CDC events to update the index."""
        events = self._log.read_from(self._position)
        count = 0
        for event in events:
            if event.table == self._source_table:
                if event.operation == Operation.DELETE:
                    self._remove_from_index(event.key)
                elif event.operation == Operation.UPDATE:
                    self._remove_from_index(event.key)
                    self._add_to_index(event.key, event.after)
                elif event.operation == Operation.INSERT:
                    self._add_to_index(event.key, event.after)
            count += 1
        self._position += count
        return count

    def search(self, keyword: str) -> list[dict]:
        """Search for rows containing the keyword in any indexed column."""
        token = keyword.lower()
        keys = self._index.get(token, set())
        return [dict(self._rows[k]) for k in keys if k in self._rows]


def create_snapshot(db: CDCDatabase, table: str) -> tuple[list[ChangeEvent], int]:
    """Create a consistent snapshot as synthetic INSERT events + current log position."""
    pos = db.cdc_log.current_position
    t = db._get_table(table)
    events = []
    for pk, row in t["rows"].items():
        events.append(ChangeEvent(
            sequence_number=-1,
            table=table,
            operation=Operation.INSERT,
            key=pk,
            before=None,
            after=dict(row),
            timestamp=datetime.now(),
        ))
    return events, pos
