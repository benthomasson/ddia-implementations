"""Vector clocks for causality tracking in distributed systems."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class VectorClock:
    """Immutable vector clock mapping node IDs to counters."""

    def __init__(self, clock: Dict[str, int] = None):
        # Strip zero entries for cleanliness
        self._clock = {k: v for k, v in (clock or {}).items() if v > 0}

    def increment(self, node_id: str) -> 'VectorClock':
        """Return a new VectorClock with node_id's counter incremented."""
        new = dict(self._clock)
        new[node_id] = new.get(node_id, 0) + 1
        return VectorClock(new)

    def merge(self, other: 'VectorClock') -> 'VectorClock':
        """Return a new VectorClock with element-wise max."""
        all_keys = set(self._clock) | set(other._clock)
        return VectorClock({k: max(self.get(k), other.get(k)) for k in all_keys})

    def get(self, node_id: str) -> int:
        """Get the counter for a node (0 if not present)."""
        return self._clock.get(node_id, 0)

    def dominates(self, other: 'VectorClock') -> bool:
        """Return True if self >= other (all entries >=)."""
        for k in set(self._clock) | set(other._clock):
            if self.get(k) < other.get(k):
                return False
        return True

    def compare(self, other: 'VectorClock') -> str:
        """Return 'BEFORE', 'AFTER', 'EQUAL', or 'CONCURRENT'."""
        has_less = False
        has_greater = False
        for k in set(self._clock) | set(other._clock):
            s, o = self.get(k), other.get(k)
            if s < o:
                has_less = True
            elif s > o:
                has_greater = True
            if has_less and has_greater:
                return "CONCURRENT"
        if has_less:
            return "BEFORE"
        if has_greater:
            return "AFTER"
        return "EQUAL"

    def is_concurrent(self, other: 'VectorClock') -> bool:
        """Return True if neither self dominates other nor vice versa."""
        return self.compare(other) == "CONCURRENT"

    def descends_from(self, other: 'VectorClock') -> bool:
        """Return True if self >= other."""
        return self.dominates(other)

    def prune(self, max_nodes: int) -> 'VectorClock':
        """Return a new VectorClock keeping only the max_nodes highest-counter entries."""
        if len(self._clock) <= max_nodes:
            return VectorClock(dict(self._clock))
        sorted_entries = sorted(self._clock.items(), key=lambda x: x[1], reverse=True)
        return VectorClock(dict(sorted_entries[:max_nodes]))

    def __eq__(self, other) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._clock == other._clock

    def __hash__(self):
        return hash(frozenset(self._clock.items()))

    def __repr__(self) -> str:
        return f"VectorClock({self._clock})"


@dataclass
class VersionedValue:
    """A value tagged with a vector clock."""
    value: str = ""
    vector_clock: VectorClock = field(default_factory=VectorClock)


@dataclass
class _HistoryEntry:
    """Internal record of a version event."""
    action: str  # "write", "sibling_created", "reconciled"
    key: str
    value: str
    vector_clock: VectorClock


class VersionedKVStore:
    """Key-value store using vector clocks for conflict detection."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._data: Dict[str, List[VersionedValue]] = {}
        self._clock = VectorClock()
        self.history: List[_HistoryEntry] = []

    def put(self, key: str, value: str, context: VectorClock = None) -> VectorClock:
        """Write a value for a key. Returns the vector clock of the new version."""
        self._clock = self._clock.increment(self.node_id)

        if context is not None:
            new_vc = context.merge(self._clock)
        else:
            new_vc = VectorClock(dict(self._clock._clock))

        existing = self._data.get(key, [])
        surviving = []
        had_concurrent = False

        for v in existing:
            if new_vc.dominates(v.vector_clock):
                pass  # new version supersedes this one
            else:
                surviving.append(v)  # concurrent sibling survives
                had_concurrent = True

        new_version = VersionedValue(value=value, vector_clock=new_vc)
        surviving.append(new_version)
        self._data[key] = surviving

        action = "sibling_created" if had_concurrent else "write"
        self.history.append(_HistoryEntry(action=action, key=key, value=value, vector_clock=new_vc))
        self._clock = self._clock.merge(new_vc)

        return new_vc

    def _receive_replica(self, key: str, value: str, vector_clock: VectorClock):
        """Receive a replicated version from another node (anti-entropy).
        Adds the version as a sibling if concurrent with existing versions,
        or replaces dominated versions."""
        existing = self._data.get(key, [])
        surviving = []
        for v in existing:
            if vector_clock.dominates(v.vector_clock):
                pass  # incoming dominates, drop existing
            else:
                surviving.append(v)
        # Only add if not dominated by any surviving version
        dominated = any(v.vector_clock.dominates(vector_clock) for v in surviving)
        if not dominated:
            surviving.append(VersionedValue(value=value, vector_clock=vector_clock))
        self._data[key] = surviving
        self._clock = self._clock.merge(vector_clock)

    def get(self, key: str) -> List[VersionedValue]:
        """Return all current versions (siblings) for a key."""
        return list(self._data.get(key, []))

    def reconcile(self, key: str, merged_value: str, contexts: List[VectorClock]) -> VectorClock:
        """Resolve conflicting siblings by merging contexts and incrementing."""
        merged_vc = VectorClock()
        for ctx in contexts:
            merged_vc = merged_vc.merge(ctx)
        merged_vc = merged_vc.merge(self._clock)
        merged_vc = merged_vc.increment(self.node_id)

        self._clock = self._clock.merge(merged_vc)

        new_version = VersionedValue(value=merged_value, vector_clock=merged_vc)
        self._data[key] = [new_version]
        self.history.append(_HistoryEntry(action="reconciled", key=key, value=merged_value, vector_clock=merged_vc))

        return merged_vc

    def keys(self) -> List[str]:
        """Return all keys in the store."""
        return list(self._data.keys())


def find_conflicts(versions: List[VersionedValue]) -> bool:
    """Return True if there are concurrent (conflicting) versions."""
    for i in range(len(versions)):
        for j in range(i + 1, len(versions)):
            if versions[i].vector_clock.is_concurrent(versions[j].vector_clock):
                return True
    return False
