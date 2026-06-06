"""SSTable format with size-tiered and leveled compaction strategies."""

import heapq
import os
import struct
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

MAGIC = b"SSTB"
VERSION = 1
TOMBSTONE_MARKER = 0xFF

# Header: magic(4) + version(2) + entry_count(4) = 10 bytes
HEADER_FMT = ">4sHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Footer: index_offset(8) + index_count(4) = 12 bytes
FOOTER_FMT = ">QI"
FOOTER_SIZE = struct.calcsize(FOOTER_FMT)


@dataclass
class SSTableEntry:
    """A key-value entry in an SSTable."""
    key: str
    value: Optional[str]  # None = tombstone
    timestamp: float


@dataclass
class SSTableMetadata:
    """Metadata about an SSTable file."""
    filepath: str
    min_key: str
    max_key: str
    entry_count: int
    file_size: int
    level: int
    timestamp: float


class SSTableWriter:
    """Build SSTables from sorted key-value pairs."""

    def __init__(self, filepath: str, block_size: int = 64):
        self._filepath = filepath
        self._block_size = block_size
        self._f = open(filepath, "wb")
        self._f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, 0))
        self._count = 0
        self._index_entries: list = []
        self._min_key: Optional[str] = None
        self._max_key: Optional[str] = None
        self._last_key: Optional[str] = None
        self._timestamp = 0.0

    def add(self, key: str, value: Optional[str], timestamp: float) -> None:
        """Add a sorted entry. Keys must be added in sorted order."""
        if self._last_key is not None and key <= self._last_key:
            raise ValueError(f"Keys must be added in sorted order: {key!r} <= {self._last_key!r}")
        self._last_key = key
        key_bytes = key.encode("utf-8")
        offset = self._f.tell()

        if self._count % self._block_size == 0:
            self._index_entries.append((key_bytes, offset))

        # Entry format: [key_len:2][key][timestamp:8][tombstone|value_len:4+value]
        self._f.write(struct.pack(">H", len(key_bytes)))
        self._f.write(key_bytes)
        self._f.write(struct.pack(">d", timestamp))

        if value is None:
            self._f.write(struct.pack("B", TOMBSTONE_MARKER))
        else:
            val_bytes = value.encode("utf-8")
            self._f.write(struct.pack(">I", len(val_bytes)))
            self._f.write(val_bytes)

        if self._min_key is None:
            self._min_key = key
        self._max_key = key
        self._timestamp = max(self._timestamp, timestamp)
        self._count += 1

    def finish(self) -> SSTableMetadata:
        """Finalize the SSTable, write index and footer. Returns metadata."""
        index_offset = self._f.tell()
        for key_bytes, offset in self._index_entries:
            self._f.write(struct.pack(">H", len(key_bytes)))
            self._f.write(key_bytes)
            self._f.write(struct.pack(">Q", offset))

        self._f.write(struct.pack(FOOTER_FMT, index_offset, len(self._index_entries)))

        self._f.seek(0)
        self._f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, self._count))
        self._f.close()

        file_size = os.path.getsize(self._filepath)
        return SSTableMetadata(
            filepath=self._filepath,
            min_key=self._min_key or "",
            max_key=self._max_key or "",
            entry_count=self._count,
            file_size=file_size,
            level=0,
            timestamp=self._timestamp,
        )


class SSTableReader:
    """Read from an SSTable file."""

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._file_size = os.path.getsize(filepath)

        with open(filepath, "rb") as f:
            magic, version, self._entry_count = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
            assert magic == MAGIC, f"Invalid magic: {magic}"
            self._data_start = HEADER_SIZE

            f.seek(-FOOTER_SIZE, 2)
            self._index_offset, index_count = struct.unpack(FOOTER_FMT, f.read(FOOTER_SIZE))

            f.seek(self._index_offset)
            self._index: list = []
            for _ in range(index_count):
                klen = struct.unpack(">H", f.read(2))[0]
                key = f.read(klen).decode("utf-8")
                offset = struct.unpack(">Q", f.read(8))[0]
                self._index.append((key, offset))

            self._min_key = ""
            self._max_key = ""
            self._timestamp = 0.0
            if self._entry_count > 0:
                f.seek(self._data_start)
                first = self._read_entry(f)
                self._min_key = first.key
                self._timestamp = first.timestamp
                if self._entry_count == 1:
                    self._max_key = first.key
                else:
                    # Scan from last index entry to find max key
                    last_offset = self._index[-1][1] if self._index else self._data_start
                    f.seek(last_offset)
                    last = None
                    while f.tell() < self._index_offset:
                        e = self._read_entry(f)
                        if e is None:
                            break
                        last = e
                        self._timestamp = max(self._timestamp, e.timestamp)
                    self._max_key = last.key if last else first.key

        self.level = 0

    @staticmethod
    def _read_entry(f) -> Optional[SSTableEntry]:
        """Read one entry from current file position."""
        data = f.read(2)
        if len(data) < 2:
            return None
        klen = struct.unpack(">H", data)[0]
        key = f.read(klen).decode("utf-8")

        ts_data = f.read(8)
        if len(ts_data) < 8:
            return None
        timestamp = struct.unpack(">d", ts_data)[0]

        marker = f.read(1)
        if len(marker) < 1:
            return None
        if marker[0] == TOMBSTONE_MARKER:
            return SSTableEntry(key=key, value=None, timestamp=timestamp)

        remaining = f.read(3)
        vlen = struct.unpack(">I", marker + remaining)[0]
        value = f.read(vlen).decode("utf-8")
        return SSTableEntry(key=key, value=value, timestamp=timestamp)

    def _iter_entries_from(self, f, start_offset) -> Iterator[SSTableEntry]:
        """Iterate entries from offset until index block."""
        f.seek(start_offset)
        while f.tell() < self._index_offset:
            entry = self._read_entry(f)
            if entry is None:
                break
            yield entry

    def get(self, key: str) -> Optional[SSTableEntry]:
        """Look up a key using sparse index + scan."""
        if self._entry_count == 0:
            return None

        # Binary search in sparse index
        lo, hi = 0, len(self._index) - 1
        block_idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._index[mid][0] <= key:
                block_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        start_offset = self._index[block_idx][1]
        end_offset = (self._index[block_idx + 1][1]
                      if block_idx + 1 < len(self._index)
                      else self._index_offset)

        with open(self._filepath, "rb") as f:
            f.seek(start_offset)
            while f.tell() < end_offset:
                entry = self._read_entry(f)
                if entry is None:
                    break
                if entry.key == key:
                    return entry
                if entry.key > key:
                    return None
        return None

    def scan(self) -> Iterator[SSTableEntry]:
        """Iterate all entries in sorted order."""
        with open(self._filepath, "rb") as f:
            yield from self._iter_entries_from(f, self._data_start)

    def range_scan(self, start_key: str, end_key: str) -> Iterator[SSTableEntry]:
        """Iterate entries where start_key <= key < end_key."""
        for entry in self.scan():
            if entry.key >= end_key:
                break
            if entry.key >= start_key:
                yield entry

    def metadata(self) -> SSTableMetadata:
        """Return metadata about this SSTable."""
        return SSTableMetadata(
            filepath=self._filepath,
            min_key=self._min_key,
            max_key=self._max_key,
            entry_count=self._entry_count,
            file_size=self._file_size,
            level=self.level,
            timestamp=self._timestamp,
        )


def merge_sstables(readers: List[SSTableReader], output_path: str,
                   remove_tombstones: bool = False) -> SSTableReader:
    """Merge multiple SSTables into one. K-way merge by key, newest timestamp wins."""
    heap: list = []
    iters: list = []

    for i, reader in enumerate(readers):
        it = iter(reader.scan())
        iters.append(it)
        try:
            entry = next(it)
            heapq.heappush(heap, (entry.key, -entry.timestamp, i, entry, it))
        except StopIteration:
            pass

    def _push_next(iterator, source_idx):
        try:
            nxt = next(iterator)
            heapq.heappush(heap, (nxt.key, -nxt.timestamp, source_idx, nxt, iterator))
        except StopIteration:
            pass

    writer = SSTableWriter(output_path)
    last_key = None

    while heap:
        key, neg_ts, idx, entry, it = heapq.heappop(heap)

        if key == last_key:
            _push_next(it, idx)
            continue

        last_key = key

        if remove_tombstones and entry.value is None:
            _push_next(it, idx)
            continue

        writer.add(entry.key, entry.value, entry.timestamp)
        _push_next(it, idx)

    writer.finish()
    return SSTableReader(output_path)


class CompactionManager:
    """Manages SSTables and runs compaction using configured strategy."""

    def __init__(self, data_dir: str, strategy: str = "size_tiered", **strategy_params):
        self._data_dir = data_dir
        self._strategy = strategy
        self._sstables: List[SSTableReader] = []
        self._counter = 0
        self._min_threshold = strategy_params.get("min_threshold", 4)
        self._size_thresholds = strategy_params.get(
            "size_thresholds", [1_000_000, 10_000_000, 100_000_000])
        self._l0_trigger = strategy_params.get("l0_compaction_trigger", 4)
        self._level_base_size = strategy_params.get("level_base_size", 10_000_000)
        self._fanout = strategy_params.get("fanout", 10)
        self._max_levels = strategy_params.get("max_levels", 7)

    def add_sstable(self, sstable: SSTableReader) -> None:
        """Register a new SSTable."""
        self._sstables.append(sstable)

    def get_sstables(self) -> List[SSTableReader]:
        """Return all current SSTables."""
        return list(self._sstables)

    def needs_compaction(self) -> bool:
        """Check if compaction should be triggered."""
        if self._strategy == "size_tiered":
            return self._stcs_needs_compaction()
        return self._lcs_needs_compaction()

    def run_compaction(self) -> List[SSTableReader]:
        """Execute one round of compaction. Returns new SSTables."""
        if self._strategy == "size_tiered":
            return self._stcs_compact()
        return self._lcs_compact()

    def _next_path(self) -> str:
        self._counter += 1
        return os.path.join(self._data_dir,
                            f"compact_{self._counter}_{int(time.time()*1000)}.sst")

    # --- Size-Tiered Compaction ---

    def _get_tier(self, size: int) -> int:
        for i, threshold in enumerate(self._size_thresholds):
            if size < threshold:
                return i
        return len(self._size_thresholds)

    def _stcs_buckets(self) -> dict:
        buckets: dict = {}
        for sst in self._sstables:
            tier = self._get_tier(sst.metadata().file_size)
            buckets.setdefault(tier, []).append(sst)
        return buckets

    def _stcs_needs_compaction(self) -> bool:
        for bucket in self._stcs_buckets().values():
            if len(bucket) >= self._min_threshold:
                return True
        return False

    def _stcs_compact(self) -> List[SSTableReader]:
        buckets = self._stcs_buckets()
        new_sstables = []
        for tier, bucket in buckets.items():
            if len(bucket) >= self._min_threshold:
                merged = merge_sstables(bucket, self._next_path())
                new_sstables.append(merged)
                for sst in bucket:
                    self._sstables.remove(sst)
        self._sstables.extend(new_sstables)
        return new_sstables

    # --- Leveled Compaction ---

    def _get_levels(self) -> dict:
        levels: dict = {}
        for sst in self._sstables:
            levels.setdefault(sst.level, []).append(sst)
        return levels

    def _level_max_size(self, level: int) -> int:
        if level == 0:
            return 0
        return int(self._level_base_size * (self._fanout ** (level - 1)))

    def _lcs_needs_compaction(self) -> bool:
        levels = self._get_levels()
        if len(levels.get(0, [])) >= self._l0_trigger:
            return True
        for lvl in range(1, self._max_levels):
            total = sum(s.metadata().file_size for s in levels.get(lvl, []))
            if total > self._level_max_size(lvl):
                return True
        return False

    def _overlapping(self, sst: SSTableReader,
                     candidates: List[SSTableReader]) -> List[SSTableReader]:
        """Find SSTables in candidates whose key range overlaps with sst."""
        meta = sst.metadata()
        return [c for c in candidates
                if c.metadata().min_key <= meta.max_key
                and c.metadata().max_key >= meta.min_key]

    def _lcs_compact(self) -> List[SSTableReader]:
        levels = self._get_levels()

        # L0 compaction
        if len(levels.get(0, [])) >= self._l0_trigger:
            l0 = levels[0]
            l1 = levels.get(1, [])
            to_merge_l1 = set()
            for sst in l0:
                for o in self._overlapping(sst, l1):
                    to_merge_l1.add(id(o))
            merge_set = list(l0) + [s for s in l1 if id(s) in to_merge_l1]
            merged = merge_sstables(merge_set, self._next_path())
            merged.level = 1
            for sst in merge_set:
                self._sstables.remove(sst)
            self._sstables.append(merged)
            return [merged]

        # Level N compaction
        for lvl in range(1, self._max_levels):
            level_ssts = levels.get(lvl, [])
            total = sum(s.metadata().file_size for s in level_ssts)
            if total > self._level_max_size(lvl):
                next_level = levels.get(lvl + 1, [])
                # Pick SSTable with most overlap with next level
                best = max(level_ssts,
                           key=lambda s: len(self._overlapping(s, next_level)))
                overlapping_next = self._overlapping(best, next_level)
                merge_set = [best] + overlapping_next
                merged = merge_sstables(merge_set, self._next_path())
                merged.level = lvl + 1
                for sst in merge_set:
                    self._sstables.remove(sst)
                self._sstables.append(merged)
                return [merged]

        return []
