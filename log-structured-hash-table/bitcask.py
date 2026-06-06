"""Log-structured hash table storage engine (Bitcask variant)."""

import os
import struct
import zlib
from typing import Optional, Iterator


TOMBSTONE = b"__BITCASK_TOMBSTONE__"
HEADER_FMT = "!III"  # crc32, key_size, value_size
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HINT_ENTRY_FMT = "!II"  # key_size, offset
HINT_HEADER_SIZE = struct.calcsize(HINT_ENTRY_FMT)


class CorruptionError(Exception):
    """Raised when data integrity check fails."""
    pass


class SegmentInfo:
    """Info about a segment file."""
    def __init__(self, segment_id: int, file_path: str, size_bytes: int,
                 num_records: int, is_active: bool):
        self.segment_id = segment_id
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.num_records = num_records
        self.is_active = is_active


class BitcaskStore:
    """Append-only log-structured key-value store with in-memory hash index."""

    def __init__(self, directory: str, max_segment_size: int = 1024 * 1024,
                 auto_compact_threshold: int = 5):
        self._dir = directory
        self._max_segment_size = max_segment_size
        self._auto_compact_threshold = auto_compact_threshold
        self._index: dict[str, tuple[str, int]] = {}  # key -> (filepath, offset)
        self._segment_counter = 0
        self._active_file = None
        self._active_path = None
        self._file_handles: dict[str, object] = {}  # path -> read file handle

        os.makedirs(directory, exist_ok=True)
        self._recover()

    def _segment_path(self, seg_id: int) -> str:
        return os.path.join(self._dir, f"segment_{seg_id:06d}.dat")

    def _hint_path(self, seg_path: str) -> str:
        return seg_path.replace(".dat", ".hint")

    def _find_existing_segments(self) -> list[tuple[int, str]]:
        """Return sorted list of (segment_id, path) for existing segment files."""
        segments = []
        for fname in os.listdir(self._dir):
            if fname.startswith("segment_") and fname.endswith(".dat"):
                seg_id = int(fname[len("segment_"):-len(".dat")])
                segments.append((seg_id, os.path.join(self._dir, fname)))
        segments.sort()
        return segments

    def _recover(self):
        """Rebuild index from existing segments on startup."""
        segments = self._find_existing_segments()
        if not segments:
            self._segment_counter = 0
            self._open_new_segment()
            return

        # Rebuild index from all segments (oldest to newest)
        for seg_id, seg_path in segments:
            hint_path = self._hint_path(seg_path)
            if os.path.exists(hint_path):
                self._load_hint_file(hint_path, seg_path)
            else:
                self._scan_segment(seg_path)

        # The highest segment id becomes the active segment
        max_id = segments[-1][0]
        self._segment_counter = max_id
        self._active_path = segments[-1][1]
        self._active_file = open(self._active_path, "ab")

    def _scan_segment(self, seg_path: str):
        """Scan a segment file and update the index."""
        with open(seg_path, "rb") as f:
            while True:
                offset = f.tell()
                header = f.read(HEADER_SIZE)
                if len(header) < HEADER_SIZE:
                    break  # end of file or partial write
                crc, key_size, value_size = struct.unpack(HEADER_FMT, header)
                payload = f.read(key_size + value_size)
                if len(payload) < key_size + value_size:
                    break  # partial write, skip
                expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if crc != expected_crc:
                    break  # corrupted record at end, stop scanning
                key = payload[:key_size].decode("utf-8")
                value = payload[key_size:]
                if value == TOMBSTONE:
                    self._index.pop(key, None)
                else:
                    self._index[key] = (seg_path, offset)

    def _load_hint_file(self, hint_path: str, seg_path: str):
        """Load index entries from a hint file."""
        with open(hint_path, "rb") as f:
            while True:
                header = f.read(HINT_HEADER_SIZE)
                if len(header) < HINT_HEADER_SIZE:
                    break
                key_size, offset = struct.unpack(HINT_ENTRY_FMT, header)
                key_bytes = f.read(key_size)
                if len(key_bytes) < key_size:
                    break
                key = key_bytes.decode("utf-8")
                self._index[key] = (seg_path, offset)

    def _open_new_segment(self):
        if self._active_file:
            self._active_file.close()
        self._active_path = self._segment_path(self._segment_counter)
        self._active_file = open(self._active_path, "ab")

    def _rotate_segment(self):
        """Close active segment and open a new one."""
        self._segment_counter += 1
        self._open_new_segment()

    def _get_read_handle(self, path: str):
        if path not in self._file_handles:
            self._file_handles[path] = open(path, "rb")
        return self._file_handles[path]

    def _write_record(self, key: str, value: bytes) -> int:
        key_bytes = key.encode("utf-8")
        payload = key_bytes + value
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FMT, crc, len(key_bytes), len(value))
        offset = self._active_file.tell()
        self._active_file.write(header + payload)
        self._active_file.flush()
        return offset

    def put(self, key: str, value: bytes) -> int:
        """Store a key-value pair. Returns the byte offset."""
        # Check if we need to rotate
        if self._active_file.tell() >= self._max_segment_size:
            self._rotate_segment()

        offset = self._write_record(key, value)
        # Close old read handle if exists for active path (stale position)
        self._file_handles.pop(self._active_path, None)
        self._index[key] = (self._active_path, offset)

        # Auto-compact check
        frozen = self._frozen_segment_paths()
        if len(frozen) >= self._auto_compact_threshold:
            self.compact()

        return offset

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a value by key. Returns None if not found."""
        if key not in self._index:
            return None
        seg_path, offset = self._index[key]
        # Use a fresh file handle for reading to avoid position conflicts
        with open(seg_path, "rb") as f:
            f.seek(offset)
            header = f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                raise CorruptionError(f"Incomplete header at offset {offset}")
            crc, key_size, value_size = struct.unpack(HEADER_FMT, header)
            payload = f.read(key_size + value_size)
            if len(payload) < key_size + value_size:
                raise CorruptionError(f"Incomplete record at offset {offset}")
            expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
            if crc != expected_crc:
                raise CorruptionError(
                    f"CRC mismatch at offset {offset}: expected {expected_crc}, got {crc}")
            return payload[key_size:]

    def delete(self, key: str) -> bool:
        """Delete a key by writing a tombstone. Returns True if key existed."""
        existed = key in self._index
        if self._active_file.tell() >= self._max_segment_size:
            self._rotate_segment()
        self._write_record(key, TOMBSTONE)
        self._index.pop(key, None)
        return existed

    def contains(self, key: str) -> bool:
        """Check if a key exists (index lookup only, no disk read)."""
        return key in self._index

    def keys(self) -> list[str]:
        """Return all live keys."""
        return list(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        for key in list(self._index.keys()):
            value = self.get(key)
            if value is not None:
                yield key, value

    def _frozen_segment_paths(self) -> list[str]:
        """Return paths of frozen (non-active) segments, sorted by id."""
        segments = self._find_existing_segments()
        return [path for _, path in segments if path != self._active_path]

    def compact(self) -> int:
        """Compact all frozen segments. Returns number of stale records removed."""
        frozen = self._frozen_segment_paths()
        if not frozen:
            return 0

        # Count total records in frozen segments
        total_records = 0
        live_entries: dict[str, tuple[str, int]] = {}

        for seg_path in frozen:
            with open(seg_path, "rb") as f:
                while True:
                    offset = f.tell()
                    header = f.read(HEADER_SIZE)
                    if len(header) < HEADER_SIZE:
                        break
                    crc, key_size, value_size = struct.unpack(HEADER_FMT, header)
                    payload = f.read(key_size + value_size)
                    if len(payload) < key_size + value_size:
                        break
                    total_records += 1
                    key = payload[:key_size].decode("utf-8")
                    value = payload[key_size:]
                    if value == TOMBSTONE:
                        live_entries.pop(key, None)
                    else:
                        live_entries[key] = (seg_path, offset)

        # Only keep entries whose index still points to a frozen segment
        entries_to_write = {}
        for key, (seg_path, offset) in live_entries.items():
            if key in self._index and self._index[key] == (seg_path, offset):
                entries_to_write[key] = (seg_path, offset)

        # Write live entries to a new compacted segment
        # Use a segment ID higher than current counter but below active
        # We'll close active, write compacted, then reopen active with higher ID
        old_active_path = self._active_path
        self._active_file.close()

        self._segment_counter += 1
        compacted_path = self._segment_path(self._segment_counter)
        new_index_entries = {}

        with open(compacted_path, "wb") as out:
            for key, (seg_path, offset) in entries_to_write.items():
                with open(seg_path, "rb") as f:
                    f.seek(offset)
                    header = f.read(HEADER_SIZE)
                    crc, key_size, value_size = struct.unpack(HEADER_FMT, header)
                    payload = f.read(key_size + value_size)
                    value = payload[key_size:]

                key_bytes = key.encode("utf-8")
                new_payload = key_bytes + value
                new_crc = zlib.crc32(new_payload) & 0xFFFFFFFF
                new_header = struct.pack(HEADER_FMT, new_crc, len(key_bytes), len(value))
                new_offset = out.tell()
                out.write(new_header + new_payload)
                new_index_entries[key] = (compacted_path, new_offset)

        # Atomically update index
        self._index.update(new_index_entries)

        # Rename active file first (atomic on POSIX), then delete old files.
        # This ordering ensures a crash always leaves valid data on disk.
        self._segment_counter += 1
        new_active_path = self._segment_path(self._segment_counter)
        os.rename(old_active_path, new_active_path)
        for key in list(self._index):
            if self._index[key][0] == old_active_path:
                self._index[key] = (new_active_path, self._index[key][1])
        self._active_path = new_active_path
        self._active_file = open(new_active_path, "ab")

        # Close cached handles and delete old frozen segments (safe: merged file is already in place)
        for seg_path in frozen:
            handle = self._file_handles.pop(seg_path, None)
            if handle:
                handle.close()
            os.remove(seg_path)
            hint = self._hint_path(seg_path)
            if os.path.exists(hint):
                os.remove(hint)

        stale_removed = total_records - len(entries_to_write)
        return stale_removed

    def close(self):
        """Flush and close all file handles."""
        if self._active_file:
            self._active_file.close()
            self._active_file = None
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()

    def segments(self) -> list[SegmentInfo]:
        """Return info about all segments."""
        result = []
        for seg_id, seg_path in self._find_existing_segments():
            size = os.path.getsize(seg_path)
            # Count records
            num_records = 0
            with open(seg_path, "rb") as f:
                while True:
                    header = f.read(HEADER_SIZE)
                    if len(header) < HEADER_SIZE:
                        break
                    _, key_size, value_size = struct.unpack(HEADER_FMT, header)
                    data = f.read(key_size + value_size)
                    if len(data) < key_size + value_size:
                        break
                    num_records += 1
            result.append(SegmentInfo(
                segment_id=seg_id, file_path=seg_path, size_bytes=size,
                num_records=num_records, is_active=(seg_path == self._active_path)))
        return result

    @property
    def total_disk_usage(self) -> int:
        total = 0
        for _, seg_path in self._find_existing_segments():
            total += os.path.getsize(seg_path)
        return total

    @property
    def num_segments(self) -> int:
        return len(self._find_existing_segments())

    def rebuild_index(self):
        """Rebuild the in-memory index by scanning all segments."""
        self._index.clear()
        for _, seg_path in self._find_existing_segments():
            hint_path = self._hint_path(seg_path)
            if os.path.exists(hint_path):
                self._load_hint_file(hint_path, seg_path)
            else:
                self._scan_segment(seg_path)

    def create_hint_files(self):
        """Write hint files for all frozen segments."""
        for seg_id, seg_path in self._find_existing_segments():
            if seg_path == self._active_path:
                continue
            hint_path = self._hint_path(seg_path)
            with open(hint_path, "wb") as hf:
                with open(seg_path, "rb") as sf:
                    while True:
                        offset = sf.tell()
                        header = sf.read(HEADER_SIZE)
                        if len(header) < HEADER_SIZE:
                            break
                        crc, key_size, value_size = struct.unpack(HEADER_FMT, header)
                        payload = sf.read(key_size + value_size)
                        if len(payload) < key_size + value_size:
                            break
                        key = payload[:key_size].decode("utf-8")
                        value = payload[key_size:]
                        if value == TOMBSTONE:
                            continue
                        # Only write hint if this key's index points here
                        if key in self._index and self._index[key] == (seg_path, offset):
                            key_bytes = key.encode("utf-8")
                            hf.write(struct.pack(HINT_ENTRY_FMT, len(key_bytes), offset))
                            hf.write(key_bytes)
