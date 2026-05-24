"""State-based CRDTs: G-Counter, PN-Counter, LWW-Register, OR-Set."""

from copy import deepcopy


class GCounter:
    """Grow-only counter using a {replica_id: count} map."""

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.counts = {}

    def increment(self, amount=1):
        """Increment this replica's counter."""
        if amount < 0:
            raise ValueError("G-Counter only supports non-negative increments")
        self.counts[self.replica_id] = self.counts.get(self.replica_id, 0) + amount

    def value(self):
        """Return the total count across all replicas."""
        return sum(self.counts.values())

    def merge(self, other):
        """Merge with another G-Counter. Mutates self and returns self."""
        for rid, count in other.counts.items():
            self.counts[rid] = max(self.counts.get(rid, 0), count)
        return self

    def state(self):
        return {"counts": dict(self.counts)}

    def __eq__(self, other):
        if not isinstance(other, GCounter):
            return NotImplemented
        return self.counts == other.counts

    def __repr__(self):
        return f"GCounter({self.replica_id}, value={self.value()})"


class PNCounter:
    """Positive-negative counter composed of two G-Counters."""

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.p = GCounter(replica_id)
        self.n = GCounter(replica_id)

    def increment(self, amount=1):
        self.p.increment(amount)

    def decrement(self, amount=1):
        self.n.increment(amount)

    def value(self):
        return self.p.value() - self.n.value()

    def merge(self, other):
        self.p.merge(other.p)
        self.n.merge(other.n)
        return self

    def state(self):
        return {"p": self.p.state(), "n": self.n.state()}

    def __eq__(self, other):
        if not isinstance(other, PNCounter):
            return NotImplemented
        return self.p == other.p and self.n == other.n

    def __repr__(self):
        return f"PNCounter({self.replica_id}, value={self.value()})"


class LWWRegister:
    """Last-writer-wins register with timestamp + replica_id tiebreaker."""

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self._value = None
        self._timestamp = 0.0
        self._writer_id = replica_id
        self._clock = 0

    def set(self, value, timestamp=None):
        """Set the register value. Uses auto-incrementing clock if no timestamp given."""
        if timestamp is None:
            self._clock += 1
            timestamp = float(self._clock)
        else:
            self._clock = max(self._clock, int(timestamp))
        self._value = value
        self._timestamp = timestamp
        self._writer_id = self.replica_id

    def get(self):
        return self._value

    def get_timestamp(self):
        return self._timestamp

    def merge(self, other):
        """Merge: highest timestamp wins. On tie, higher replica_id wins."""
        if (other._timestamp, other._writer_id) > (self._timestamp, self._writer_id):
            self._value = other._value
            self._timestamp = other._timestamp
            self._writer_id = other._writer_id
        self._clock = max(self._clock, other._clock)
        return self

    def state(self):
        return {
            "value": self._value,
            "timestamp": self._timestamp,
            "writer_id": self._writer_id,
        }

    def __eq__(self, other):
        if not isinstance(other, LWWRegister):
            return NotImplemented
        return (self._value == other._value and
                self._timestamp == other._timestamp and
                self._writer_id == other._writer_id)

    def __repr__(self):
        return f"LWWRegister({self.replica_id}, value={self._value!r}, ts={self._timestamp})"


class ORSet:
    """Observed-remove set with unique tags and tombstones."""

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self._seq = 0
        # {element: set of tags} where tag = (replica_id, seq_num)
        self._elements = {}
        # set of removed tags
        self._tombstones = set()

    def _new_tag(self):
        self._seq += 1
        return (self.replica_id, self._seq)

    def add(self, element):
        """Add an element with a new unique tag."""
        tag = self._new_tag()
        if element not in self._elements:
            self._elements[element] = set()
        self._elements[element].add(tag)

    def remove(self, element):
        """Remove all known tags for an element. Returns True if element was present."""
        if element not in self._elements or not self._elements[element]:
            return False
        # Move all current tags to tombstones
        self._tombstones.update(self._elements[element])
        del self._elements[element]
        return True

    def contains(self, element):
        return element in self._elements and len(self._elements[element]) > 0

    def elements(self):
        return {e for e, tags in self._elements.items() if tags}

    def merge(self, other):
        """Merge OR-Sets. For each element, keep tags in either active set minus both tombstone sets."""
        all_elements = set(self._elements.keys()) | set(other._elements.keys())
        merged_tombstones = self._tombstones | other._tombstones

        new_elements = {}
        for elem in all_elements:
            self_tags = self._elements.get(elem, set())
            other_tags = other._elements.get(elem, set())
            # Union of active tags, minus anything tombstoned by either side
            merged_tags = (self_tags | other_tags) - merged_tombstones
            if merged_tags:
                new_elements[elem] = merged_tags

        self._elements = new_elements
        self._tombstones = merged_tombstones
        # Advance sequence counter to avoid tag collisions
        self._seq = max(self._seq, other._seq)
        return self

    def state(self):
        return {
            "elements": {str(e): [list(t) for t in tags] for e, tags in self._elements.items()},
            "tombstones": [list(t) for t in self._tombstones],
        }

    def __eq__(self, other):
        if not isinstance(other, ORSet):
            return NotImplemented
        return self._elements == other._elements and self._tombstones == other._tombstones

    def __repr__(self):
        return f"ORSet({self.replica_id}, elements={self.elements()})"


class CRDTReplicaGroup:
    """Manages multiple CRDT replicas and simulates network sync."""

    def __init__(self, crdt_class, replica_ids, **kwargs):
        self.replicas = {rid: crdt_class(rid, **kwargs) for rid in replica_ids}
        self.replica_ids = list(replica_ids)

    def get_replica(self, replica_id):
        return self.replicas[replica_id]

    def sync(self, from_id, to_id):
        """One-directional sync: merge from_id's state into to_id."""
        source = deepcopy(self.replicas[from_id])
        self.replicas[to_id].merge(source)

    def sync_all(self):
        """Full mesh sync until convergence."""
        # Two rounds of all-pairs sync guarantees convergence
        for _ in range(2):
            for a in self.replica_ids:
                for b in self.replica_ids:
                    if a != b:
                        self.sync(a, b)

    def all_converged(self):
        """Check if all replicas have the same state."""
        if len(self.replica_ids) < 2:
            return True
        first = self.replicas[self.replica_ids[0]]
        return all(self.replicas[rid] == first for rid in self.replica_ids[1:])

    def values(self):
        """Return each replica's current value."""
        result = {}
        for rid, replica in self.replicas.items():
            if hasattr(replica, 'value'):
                result[rid] = replica.value()
            elif hasattr(replica, 'get'):
                result[rid] = replica.get()
            elif hasattr(replica, 'elements'):
                result[rid] = replica.elements()
        return result
