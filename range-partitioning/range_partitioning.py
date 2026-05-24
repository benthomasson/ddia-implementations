"""Range-based partitioning with dynamic split and merge."""

import bisect
import uuid
from dataclasses import dataclass
from typing import Any, Optional


class Partition:
    """A partition storing key-value pairs in sorted order within [start_key, end_key)."""

    def __init__(self, partition_id: str, start_key: str, end_key: Optional[str]):
        self.partition_id = partition_id
        self.start_key = start_key
        self.end_key = end_key
        self._keys: list[str] = []
        self._values: list[Any] = []

    def put(self, key: str, value: Any):
        """Store a key-value pair."""
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            self._values[idx] = value
        else:
            self._keys.insert(idx, key)
            self._values.insert(idx, value)

    def get(self, key: str) -> Optional[Any]:
        """Get a value by key."""
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return self._values[idx]
        return None

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            self._keys.pop(idx)
            self._values.pop(idx)
            return True
        return False

    def range_scan(self, start: str, end: Optional[str]) -> list[tuple[str, Any]]:
        """Return all (key, value) pairs where start <= key < end, sorted."""
        left = bisect.bisect_left(self._keys, start)
        if end is None:
            right = len(self._keys)
        else:
            right = bisect.bisect_left(self._keys, end)
        return list(zip(self._keys[left:right], self._values[left:right]))

    def contains_key(self, key: str) -> bool:
        """Check if a key falls within this partition's range."""
        if key < self.start_key:
            return False
        if self.end_key is not None and key >= self.end_key:
            return False
        return True

    def split(self) -> 'Partition':
        """Split at median key. Returns the new right partition."""
        mid = len(self._keys) // 2
        median_key = self._keys[mid]

        new_partition = Partition(str(uuid.uuid4()), median_key, self.end_key)
        new_partition._keys = self._keys[mid:]
        new_partition._values = self._values[mid:]

        self._keys = self._keys[:mid]
        self._values = self._values[:mid]
        self.end_key = median_key

        return new_partition

    def merge(self, other: 'Partition'):
        """Merge another adjacent partition into this one."""
        self._keys.extend(other._keys)
        self._values.extend(other._values)
        self.end_key = other.end_key

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def key_range(self) -> tuple[str, Optional[str]]:
        return (self.start_key, self.end_key)


@dataclass
class PartitionInfo:
    partition_id: str
    start_key: str
    end_key: Optional[str]
    size: int


class RangePartitionedStore:
    """Manages multiple partitions and routes operations by key range."""

    def __init__(self, max_partition_size: int = 1000, min_partition_size: int = 100):
        self.max_partition_size = max_partition_size
        self.min_partition_size = min_partition_size
        initial = Partition(str(uuid.uuid4()), "", None)
        self._partitions: list[Partition] = [initial]
        # Sorted list of start keys for binary search routing
        self._boundaries: list[str] = [""]

    def _find_partition_index(self, key: str) -> int:
        """Find the index of the partition that owns key using binary search."""
        idx = bisect.bisect_right(self._boundaries, key) - 1
        return idx

    def put(self, key: str, value: Any):
        """Insert or update a key-value pair. Triggers auto-split if needed."""
        idx = self._find_partition_index(key)
        partition = self._partitions[idx]
        partition.put(key, value)

        if partition.size > self.max_partition_size:
            new_right = partition.split()
            self._partitions.insert(idx + 1, new_right)
            self._boundaries.insert(idx + 1, new_right.start_key)

    def get(self, key: str) -> Optional[Any]:
        """Look up a value by key."""
        idx = self._find_partition_index(key)
        return self._partitions[idx].get(key)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        idx = self._find_partition_index(key)
        return self._partitions[idx].delete(key)

    def range_scan(self, start_key: str, end_key: Optional[str] = None) -> list[tuple[str, Any]]:
        """Scan keys in [start_key, end_key) across all relevant partitions."""
        start_idx = self._find_partition_index(start_key)

        if end_key is None:
            end_idx = len(self._partitions) - 1
        else:
            end_idx = self._find_partition_index(end_key)
            # If end_key exactly equals a boundary, the previous partition is the last relevant one
            if end_idx > 0 and self._boundaries[end_idx] == end_key:
                end_idx -= 1

        results = []
        for i in range(start_idx, end_idx + 1):
            results.extend(self._partitions[i].range_scan(start_key, end_key))
        return results

    def merge_small_partitions(self) -> int:
        """Merge adjacent small partitions. Returns number of merges performed."""
        merges = 0
        i = 0
        while i < len(self._partitions) - 1:
            left = self._partitions[i]
            right = self._partitions[i + 1]
            combined = left.size + right.size
            if combined <= self.min_partition_size:
                left.merge(right)
                self._partitions.pop(i + 1)
                self._boundaries.pop(i + 1)
                merges += 1
                # Don't increment i — check if we can merge again with the next
            else:
                i += 1
        return merges

    @property
    def partition_count(self) -> int:
        return len(self._partitions)

    @property
    def total_keys(self) -> int:
        return sum(p.size for p in self._partitions)

    def get_partition_info(self) -> list[PartitionInfo]:
        """Return metadata about all partitions in key order."""
        return [
            PartitionInfo(p.partition_id, p.start_key, p.end_key, p.size)
            for p in self._partitions
        ]

    def get_partition_for_key(self, key: str) -> PartitionInfo:
        """Return the partition that owns a given key."""
        idx = self._find_partition_index(key)
        p = self._partitions[idx]
        return PartitionInfo(p.partition_id, p.start_key, p.end_key, p.size)
