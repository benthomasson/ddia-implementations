"""LSM Tree storage engine with memtable, SSTables, WAL, and compaction."""

import os
import struct
import bisect
import heapq
import zlib
from typing import Optional, List, Tuple
from sortedcontainers import SortedDict

TOMBSTONE = b""


class WAL:
    """Write-ahead log for crash recovery."""

    def __init__(self, path: str):
        self._path = path
        self._fd = open(path, "ab")

    def append(self, key: str, value: bytes):
        k = key.encode("utf-8")
        payload = struct.pack(">I", len(k)) + k + struct.pack(">I", len(value)) + value
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        self._fd.write(struct.pack(">I", crc))
        self._fd.write(payload)
        self._fd.flush()

    def replay(self) -> List[Tuple[str, bytes]]:
        """Replay WAL entries. Returns list of (key, value_bytes)."""
        entries = []
        if not os.path.exists(self._path):
            return entries
        with open(self._path, "rb") as f:
            data = f.read()
        pos = 0
        while pos < len(data):
            if pos + 4 > len(data):
                break
            stored_crc = struct.unpack(">I", data[pos:pos + 4])[0]
            record_start = pos + 4
            if record_start + 4 > len(data):
                break
            klen = struct.unpack(">I", data[record_start:record_start + 4])[0]
            kstart = record_start + 4
            if kstart + klen > len(data):
                break
            key = data[kstart:kstart + klen].decode("utf-8")
            vlen_start = kstart + klen
            if vlen_start + 4 > len(data):
                break
            vlen = struct.unpack(">I", data[vlen_start:vlen_start + 4])[0]
            vstart = vlen_start + 4
            if vstart + vlen > len(data):
                break
            value = data[vstart:vstart + vlen]
            payload = data[record_start:vstart + vlen]
            expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
            if stored_crc != expected_crc:
                break
            pos = vstart + vlen
            entries.append((key, value))
        return entries

    def truncate(self):
        self._fd.close()
        self._fd = open(self._path, "wb")
        self._fd.close()
        self._fd = open(self._path, "ab")

    def close(self):
        self._fd.close()


class SSTable:
    """Sorted String Table with sparse index."""

    def __init__(self, path: str, seq: int, sparse_index_interval: int = 16):
        self.path = path
        self.seq = seq
        self._interval = sparse_index_interval
        self._sparse_index: List[Tuple[str, int]] = []  # (key, file_offset)

    @staticmethod
    def write(path: str, seq: int, entries: List[Tuple[str, bytes]],
              sparse_index_interval: int = 16) -> "SSTable":
        """Write sorted entries to an SSTable file. Returns the SSTable."""
        sst = SSTable(path, seq, sparse_index_interval)
        with open(path, "wb") as f:
            offsets: List[Tuple[str, int]] = []
            for i, (key, value) in enumerate(entries):
                offset = f.tell()
                if i % sparse_index_interval == 0:
                    offsets.append((key, offset))
                k = key.encode("utf-8")
                f.write(struct.pack(">I", len(k)))
                f.write(k)
                f.write(struct.pack(">I", len(value)))
                f.write(value)
            # Write footer: sparse index entries then count
            footer_start = f.tell()
            for key, off in offsets:
                k = key.encode("utf-8")
                f.write(struct.pack(">I", len(k)))
                f.write(k)
                f.write(struct.pack(">Q", off))
            f.write(struct.pack(">Q", footer_start))
            f.write(struct.pack(">I", len(offsets)))
        sst._sparse_index = offsets
        return sst

    def load_index(self):
        """Load sparse index from file footer."""
        with open(self.path, "rb") as f:
            f.seek(-12, 2)  # last 12 bytes: footer_start(8) + count(4)
            footer_start = struct.unpack(">Q", f.read(8))[0]
            count = struct.unpack(">I", f.read(4))[0]
            f.seek(footer_start)
            self._sparse_index = []
            for _ in range(count):
                klen = struct.unpack(">I", f.read(4))[0]
                key = f.read(klen).decode("utf-8")
                off = struct.unpack(">Q", f.read(8))[0]
                self._sparse_index.append((key, off))

    def get(self, key: str) -> Tuple[bool, Optional[bytes]]:
        """Look up key. Returns (found, value_bytes). value_bytes is TOMBSTONE if deleted."""
        if not self._sparse_index:
            return False, None
        keys = [k for k, _ in self._sparse_index]
        idx = bisect.bisect_right(keys, key) - 1
        if idx < 0:
            idx = 0
        start_off = self._sparse_index[idx][1]
        end_off = (self._sparse_index[idx + 1][1]
                   if idx + 1 < len(self._sparse_index)
                   else self._footer_start())
        return self._scan_range_for_key(key, start_off, end_off)

    def _footer_start(self) -> int:
        with open(self.path, "rb") as f:
            f.seek(-12, 2)
            return struct.unpack(">Q", f.read(8))[0]

    def _scan_range_for_key(self, key: str, start: int, end: int) -> Tuple[bool, Optional[bytes]]:
        with open(self.path, "rb") as f:
            f.seek(start)
            while f.tell() < end:
                entry = self._read_entry(f)
                if entry is None:
                    break
                k, v = entry
                if k == key:
                    return True, v
                if k > key:
                    break
        return False, None

    def scan(self, start_key: str, end_key: str):
        """Yield (key, value_bytes) for keys in [start_key, end_key)."""
        footer = self._footer_start()
        with open(self.path, "rb") as f:
            # Find starting position via sparse index
            keys = [k for k, _ in self._sparse_index]
            idx = bisect.bisect_right(keys, start_key) - 1
            if idx < 0:
                idx = 0
            f.seek(self._sparse_index[idx][1])
            while f.tell() < footer:
                entry = self._read_entry(f)
                if entry is None:
                    break
                k, v = entry
                if k >= end_key:
                    break
                if k >= start_key:
                    yield k, v

    def scan_all(self):
        """Yield all (key, value_bytes) entries."""
        footer = self._footer_start()
        with open(self.path, "rb") as f:
            while f.tell() < footer:
                entry = self._read_entry(f)
                if entry is None:
                    break
                yield entry

    @staticmethod
    def _read_entry(f) -> Optional[Tuple[str, bytes]]:
        hdr = f.read(4)
        if len(hdr) < 4:
            return None
        klen = struct.unpack(">I", hdr)[0]
        k = f.read(klen)
        if len(k) < klen:
            return None
        vhdr = f.read(4)
        if len(vhdr) < 4:
            return None
        vlen = struct.unpack(">I", vhdr)[0]
        v = f.read(vlen)
        if len(v) < vlen:
            return None
        return k.decode("utf-8"), v


class LSMTree:
    """LSM Tree storage engine."""

    def __init__(self, data_dir: str, memtable_threshold: int = 1000,
                 sparse_index_interval: int = 16,
                 compaction_threshold: int = 4):
        self._dir = data_dir
        self._threshold = memtable_threshold
        self._sparse_interval = sparse_index_interval
        self._compaction_threshold = compaction_threshold
        self._memtable: SortedDict = SortedDict()
        self._immutable_memtables: List[SortedDict] = []
        self._sstables: List[SSTable] = []  # newest last
        self._seq = 0
        os.makedirs(data_dir, exist_ok=True)
        self._load_existing_sstables()
        self._wal = WAL(os.path.join(data_dir, "wal.log"))
        self._replay_wal()

    def _load_existing_sstables(self):
        files = sorted(f for f in os.listdir(self._dir) if f.endswith(".sst"))
        for fname in files:
            seq = int(fname.split("_")[1].split(".")[0])
            sst = SSTable(os.path.join(self._dir, fname), seq, self._sparse_interval)
            sst.load_index()
            self._sstables.append(sst)
            if seq >= self._seq:
                self._seq = seq + 1
        # Sort by sequence number (oldest first, newest last)
        self._sstables.sort(key=lambda s: s.seq)

    def _replay_wal(self):
        for key, value in self._wal.replay():
            self._memtable[key] = value

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def put(self, key: str, value: str) -> None:
        """Insert or update a key-value pair."""
        v = value.encode("utf-8")
        self._wal.append(key, v)
        self._memtable[key] = v
        if len(self._memtable) >= self._threshold:
            self._flush()

    def get(self, key: str) -> Optional[str]:
        """Retrieve value for key, or None if not found/deleted."""
        # Check active memtable
        if key in self._memtable:
            v = self._memtable[key]
            return None if v == TOMBSTONE else v.decode("utf-8")
        # Check immutable memtables (newest first)
        for mt in reversed(self._immutable_memtables):
            if key in mt:
                v = mt[key]
                return None if v == TOMBSTONE else v.decode("utf-8")
        # Check SSTables newest to oldest
        for sst in reversed(self._sstables):
            found, v = sst.get(key)
            if found:
                return None if v == TOMBSTONE else v.decode("utf-8")
        return None

    def delete(self, key: str) -> None:
        """Delete a key by writing a tombstone."""
        self._wal.append(key, TOMBSTONE)
        self._memtable[key] = TOMBSTONE
        if len(self._memtable) >= self._threshold:
            self._flush()

    def range_scan(self, start_key: str, end_key: str) -> List[Tuple[str, str]]:
        """Return sorted key-value pairs where start_key <= key < end_key."""
        # Collect iterators with priority (higher = newer)
        # We'll do a merge, keeping newest value per key
        merged: dict = {}  # key -> (priority, value_bytes)
        priority = 0

        # SSTables oldest to newest
        for sst in self._sstables:
            for k, v in sst.scan(start_key, end_key):
                merged[k] = (priority, v)
            priority += 1

        # Immutable memtables oldest to newest
        for mt in self._immutable_memtables:
            for k in mt.irange(start_key, end_key, (True, False)):
                merged[k] = (priority, mt[k])
            priority += 1

        # Active memtable (newest)
        for k in self._memtable.irange(start_key, end_key, (True, False)):
            merged[k] = (priority, self._memtable[k])

        # Build result, excluding tombstones
        result = []
        for k in sorted(merged):
            _, v = merged[k]
            if v != TOMBSTONE:
                result.append((k, v.decode("utf-8")))
        return result

    def _flush(self):
        """Flush current memtable to an SSTable."""
        if not self._memtable:
            return
        frozen = self._memtable
        seq = self._next_seq()
        path = os.path.join(self._dir, f"sst_{seq:06d}.sst")
        entries = list(frozen.items())
        sst = SSTable.write(path, seq, entries, self._sparse_interval)
        self._sstables.append(sst)
        self._wal.truncate()
        self._memtable = SortedDict()
        # Auto-compact if threshold exceeded
        if len(self._sstables) >= self._compaction_threshold:
            self.compact()

    def compact(self) -> None:
        """Merge all SSTables into one, removing tombstones."""
        if len(self._sstables) < 2:
            return
        # k-way merge: iterate all SSTables, newest wins
        # Collect all entries keyed by (key, -seq) for proper ordering
        all_entries: List[Tuple[str, int, bytes]] = []
        for sst in self._sstables:
            for k, v in sst.scan_all():
                all_entries.append((k, sst.seq, v))

        # Sort by key, then by seq descending (newest first)
        all_entries.sort(key=lambda x: (x[0], -x[1]))

        # Deduplicate: keep first occurrence of each key (newest)
        merged: List[Tuple[str, bytes]] = []
        prev_key = None
        for k, seq, v in all_entries:
            if k == prev_key:
                continue
            prev_key = k
            # Remove tombstones during compaction
            if v != TOMBSTONE:
                merged.append((k, v))

        # Write new SSTable
        seq = self._next_seq()
        path = os.path.join(self._dir, f"sst_{seq:06d}.sst")
        new_sst = SSTable.write(path, seq, merged, self._sparse_interval)

        # Delete old SSTables
        old_sstables = self._sstables
        self._sstables = [new_sst]
        for sst in old_sstables:
            os.remove(sst.path)

    def close(self) -> None:
        """Flush memtable and close WAL."""
        self._flush()
        self._wal.close()
