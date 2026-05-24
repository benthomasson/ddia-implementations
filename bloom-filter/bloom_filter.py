"""Bloom filter implementations: standard, counting, and scalable."""

import hashlib
import math
import struct


def _hashes(item, k, m):
    """Compute k hash positions using double hashing with MD5."""
    digest = hashlib.md5(item.encode("utf-8")).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little")
    return [(h1 + i * h2) % m for i in range(k)]


class BloomFilter:
    """Space-efficient probabilistic set membership filter."""

    def __init__(self, expected_items=1000, false_positive_rate=0.01,
                 bit_size=None, num_hashes=None):
        if bit_size is not None and num_hashes is not None:
            self._m = bit_size
            self._k = num_hashes
        else:
            ln2 = math.log(2)
            self._m = max(1, math.ceil(-expected_items * math.log(false_positive_rate) / (ln2 ** 2)))
            self._k = max(1, round((self._m / expected_items) * ln2))
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0

    def add(self, item):
        """Add an item to the filter."""
        for pos in _hashes(item, self._k, self._m):
            self._bits[pos // 8] |= 1 << (pos % 8)
        self._count += 1

    def __contains__(self, item):
        for pos in _hashes(item, self._k, self._m):
            if not (self._bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def __len__(self):
        return self._count

    @property
    def bit_count(self):
        return self._m

    @property
    def hash_count(self):
        return self._k

    def _set_bits_count(self):
        return sum(bin(b).count("1") for b in self._bits)

    def bit_array_density(self):
        """Fraction of bits that are set."""
        return self._set_bits_count() / self._m

    def estimated_false_positive_rate(self):
        """Current estimated FPR based on bit density."""
        return self.bit_array_density() ** self._k

    def estimate_count(self):
        """Estimate distinct items from bit density."""
        x = self._set_bits_count()
        if x == 0:
            return 0.0
        if x >= self._m:
            return float("inf")
        return -(self._m / self._k) * math.log(1 - x / self._m)

    def union(self, other):
        """Return a new filter that is the bitwise OR of self and other."""
        if self._m != other._m or self._k != other._k:
            raise ValueError("Filters must have same m and k for union")
        result = BloomFilter(bit_size=self._m, num_hashes=self._k)
        result._bits = bytearray(a | b for a, b in zip(self._bits, other._bits))
        result._count = self._count + other._count
        return result

    def to_bytes(self):
        """Serialize the filter to bytes."""
        header = struct.pack("<III", self._m, self._k, self._count)
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data):
        """Deserialize a filter from bytes."""
        m, k, count = struct.unpack("<III", data[:12])
        bf = cls(bit_size=m, num_hashes=k)
        bf._bits = bytearray(data[12:])
        bf._count = count
        return bf


class CountingBloomFilter:
    """Bloom filter with counters supporting deletion."""

    def __init__(self, expected_items=1000, false_positive_rate=0.01,
                 counter_bits=4):
        ln2 = math.log(2)
        self._m = max(1, math.ceil(-expected_items * math.log(false_positive_rate) / (ln2 ** 2)))
        self._k = max(1, round((self._m / expected_items) * ln2))
        self._max_val = (1 << counter_bits) - 1
        self._counters = bytearray(self._m)  # one byte per counter for simplicity
        self._count = 0

    def add(self, item):
        """Add an item (increment counters, saturate at max)."""
        for pos in _hashes(item, self._k, self._m):
            if self._counters[pos] < self._max_val:
                self._counters[pos] += 1
        self._count += 1

    def remove(self, item):
        """Remove an item (decrement counters). Raises ValueError if not present."""
        positions = _hashes(item, self._k, self._m)
        for pos in positions:
            if self._counters[pos] == 0:
                raise ValueError("Item was not added to the filter")
        for pos in positions:
            if self._counters[pos] < self._max_val:  # saturated counters stay
                self._counters[pos] -= 1
        self._count -= 1

    def __contains__(self, item):
        return all(self._counters[pos] > 0 for pos in _hashes(item, self._k, self._m))

    def __len__(self):
        return self._count

    @property
    def bit_count(self):
        return self._m

    @property
    def hash_count(self):
        return self._k


class ScalableBloomFilter:
    """Automatically grows by adding filter slices to maintain target FPR."""

    def __init__(self, initial_capacity=1000, false_positive_rate=0.01,
                 growth_factor=2, fp_ratio=0.5):
        self._initial_cap = initial_capacity
        self._p = false_positive_rate
        self._growth = growth_factor
        self._ratio = fp_ratio
        self._slices = []
        self._count = 0
        self._add_slice()

    def _add_slice(self):
        idx = len(self._slices)
        cap = self._initial_cap * (self._growth ** idx)
        p = self._p * (self._ratio ** idx)
        self._slices.append((BloomFilter(expected_items=int(cap), false_positive_rate=p), int(cap)))

    def add(self, item):
        """Add an item, creating a new slice if current one is full."""
        if item in self:
            return
        current, cap = self._slices[-1]
        if len(current) >= cap:
            self._add_slice()
            current, cap = self._slices[-1]
        current.add(item)
        self._count += 1

    def __contains__(self, item):
        return any(item in bf for bf, _ in self._slices)

    def __len__(self):
        return self._count
