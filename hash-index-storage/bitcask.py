"""Bitcask-style hash index storage engine."""

import os
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Optional


HEADER_FORMAT = "<IdII"  # crc32(uint32), timestamp(double), key_size(uint32), value_size(uint32)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 20 bytes

HINT_FORMAT = "<IQId"  # file_id(uint32), offset(uint64), size(uint32), timestamp(double)
HINT_HEADER_SIZE = struct.calcsize(HINT_FORMAT)


@dataclass
class KeyEntry:
    """In-memory index entry pointing to a record on disk."""
    file_id: int
    offset: int
    size: int
    timestamp: float


class BitcaskStore:
    """Bitcask-style append-only key-value store with in-memory hash index."""

    def __init__(self, data_dir: str, max_file_size: int = 10 * 1024 * 1024,
                 sync_writes: bool = True):
        """Initialize or open an existing Bitcask store."""
        self.data_dir = data_dir
        self.max_file_size = max_file_size
        self.sync_writes = sync_writes
        self.keydir: dict[str, KeyEntry] = {}
        self.file_handles: dict[int, object] = {}

        os.makedirs(data_dir, exist_ok=True)

        # Find existing data files
        existing_ids = self._find_file_ids()

        if existing_ids:
            self._rebuild_index(existing_ids)
            self.active_file_id = max(existing_ids)
        else:
            self.active_file_id = 0

        # Open active file for appending
        self._open_active_file()

    def _find_file_ids(self) -> list[int]:
        """Find all data file IDs in the data directory."""
        ids = []
        for name in os.listdir(self.data_dir):
            if name.endswith(".data"):
                ids.append(int(name[:-5]))
        return sorted(ids)

    def _data_path(self, file_id: int) -> str:
        return os.path.join(self.data_dir, f"{file_id}.data")

    def _hint_path(self, file_id: int) -> str:
        return os.path.join(self.data_dir, f"{file_id}.hint")

    def _open_active_file(self):
        path = self._data_path(self.active_file_id)
        self.active_file = open(path, "ab")
        self.active_file.seek(0, 2)  # ensure tell() returns correct position
        self.file_handles[self.active_file_id] = open(path, "rb")

    def _get_reader(self, file_id: int):
        if file_id not in self.file_handles:
            self.file_handles[file_id] = open(self._data_path(file_id), "rb")
        return self.file_handles[file_id]

    def _write_record(self, key: str, value: str) -> tuple[int, int, float]:
        """Append a record to the active file. Returns (offset, size, timestamp)."""
        ts = time.time()
        key_bytes = key.encode("utf-8")
        val_bytes = value.encode("utf-8")
        payload = key_bytes + val_bytes
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FORMAT, crc, ts, len(key_bytes), len(val_bytes))
        record = header + payload
        offset = self.active_file.tell()
        self.active_file.write(record)
        self.active_file.flush()
        if self.sync_writes:
            os.fsync(self.active_file.fileno())
        return offset, len(record), ts

    def _read_record(self, file_id: int, offset: int, size: int) -> tuple[str, str, float]:
        """Read a record from a data file. Returns (key, value, timestamp)."""
        reader = self._get_reader(file_id)
        reader.seek(offset)
        data = reader.read(size)
        crc, ts, key_size, val_size = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        payload = data[HEADER_SIZE:HEADER_SIZE + key_size + val_size]
        expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != expected_crc:
            raise IOError(f"CRC mismatch at file {file_id} offset {offset}")
        key = payload[:key_size].decode("utf-8")
        value = payload[key_size:].decode("utf-8")
        return key, value, ts

    def _maybe_rotate(self):
        """Rotate to a new active file if current one exceeds size limit."""
        if self.active_file.tell() >= self.max_file_size:
            self.active_file.close()
            self.active_file_id += 1
            self._open_active_file()

    def _rebuild_index(self, file_ids: list[int]):
        """Rebuild the in-memory index from data/hint files."""
        for fid in sorted(file_ids):
            hint_path = self._hint_path(fid)
            if os.path.exists(hint_path):
                self._load_hint_file(fid)
            else:
                self._scan_data_file(fid)

    def _scan_data_file(self, file_id: int):
        """Scan a data file to rebuild index entries."""
        path = self._data_path(file_id)
        reader = self._get_reader(file_id)
        reader.seek(0)
        file_size = os.path.getsize(path)

        while reader.tell() < file_size:
            offset = reader.tell()
            header_data = reader.read(HEADER_SIZE)
            if len(header_data) < HEADER_SIZE:
                break
            crc, ts, key_size, val_size = struct.unpack(HEADER_FORMAT, header_data)
            payload = reader.read(key_size + val_size)
            if len(payload) < key_size + val_size:
                break
            expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
            if crc != expected_crc:
                break
            key = payload[:key_size].decode("utf-8")
            record_size = HEADER_SIZE + key_size + val_size

            if val_size == 0:
                self.keydir.pop(key, None)
            else:
                self.keydir[key] = KeyEntry(file_id, offset, record_size, ts)

    def _load_hint_file(self, file_id: int):
        """Load index entries from a hint file."""
        with open(self._hint_path(file_id), "rb") as f:
            data = f.read()

        pos = 0
        while pos < len(data):
            fid, offset, size, ts = struct.unpack(HINT_FORMAT, data[pos:pos + HINT_HEADER_SIZE])
            pos += HINT_HEADER_SIZE
            # Read key_size (4 bytes) then key
            key_size = struct.unpack("<I", data[pos:pos + 4])[0]
            pos += 4
            key = data[pos:pos + key_size].decode("utf-8")
            pos += key_size
            self.keydir[key] = KeyEntry(fid, offset, size, ts)

    def _write_hint_file(self, file_id: int, entries: list[tuple[str, KeyEntry]]):
        """Write a hint file for fast index rebuild."""
        with open(self._hint_path(file_id), "wb") as f:
            for key, entry in entries:
                key_bytes = key.encode("utf-8")
                f.write(struct.pack(HINT_FORMAT, entry.file_id, entry.offset,
                                    entry.size, entry.timestamp))
                f.write(struct.pack("<I", len(key_bytes)))
                f.write(key_bytes)

    def put(self, key: str, value: str) -> None:
        """Store a key-value pair."""
        self._maybe_rotate()
        offset, size, ts = self._write_record(key, value)
        self.keydir[key] = KeyEntry(self.active_file_id, offset, size, ts)

    def get(self, key: str) -> Optional[str]:
        """Retrieve the value for a key, or None if not found/deleted."""
        entry = self.keydir.get(key)
        if entry is None:
            return None
        read_key, value, _ = self._read_record(entry.file_id, entry.offset, entry.size)
        assert read_key == key, f"Key mismatch: expected {key!r}, got {read_key!r}"
        if value == "":
            return None
        return value

    def delete(self, key: str) -> None:
        """Delete a key by appending a tombstone record."""
        self._maybe_rotate()
        self._write_record(key, "")
        self.keydir.pop(key, None)

    def keys(self) -> list[str]:
        """Return all live keys."""
        return list(self.keydir.keys())

    def compact(self) -> None:
        """Merge old data files, remove stale entries and tombstones."""
        # Only compact immutable files (not the active one)
        all_ids = self._find_file_ids()
        immutable_ids = [fid for fid in all_ids if fid != self.active_file_id]
        if not immutable_ids:
            return

        # Collect latest record per key from immutable files
        latest: dict[str, tuple[int, int, int, float]] = {}  # key -> (file_id, offset, size, ts)
        for fid in sorted(immutable_ids):
            path = self._data_path(fid)
            reader = self._get_reader(fid)
            reader.seek(0)
            file_size = os.path.getsize(path)

            while reader.tell() < file_size:
                offset = reader.tell()
                header_data = reader.read(HEADER_SIZE)
                if len(header_data) < HEADER_SIZE:
                    break
                crc, ts, key_size, val_size = struct.unpack(HEADER_FORMAT, header_data)
                payload = reader.read(key_size + val_size)
                if len(payload) < key_size + val_size:
                    break
                expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if crc != expected_crc:
                    break
                key = payload[:key_size].decode("utf-8")
                record_size = HEADER_SIZE + key_size + val_size

                if val_size == 0:
                    latest.pop(key, None)
                else:
                    if key not in latest or ts > latest[key][3]:
                        latest[key] = (fid, offset, record_size, ts)

        # Also filter: only keep keys whose latest value is in an immutable file
        # (if keydir points to active file, that key's immutable records are stale)
        keys_to_compact = {}
        for key, (fid, offset, size, ts) in latest.items():
            entry = self.keydir.get(key)
            if entry and entry.file_id in immutable_ids:
                keys_to_compact[key] = (fid, offset, size, ts)

        # Close readers for immutable files
        for fid in immutable_ids:
            if fid in self.file_handles:
                self.file_handles[fid].close()
                del self.file_handles[fid]

        # Pick new file IDs that don't conflict
        new_base_id = self.active_file_id + 1

        # Write merged data to new file(s)
        merged_file_id = new_base_id
        merged_file = open(self._data_path(merged_file_id), "ab")
        hint_entries: list[tuple[str, KeyEntry]] = []

        for key, (old_fid, old_offset, old_size, ts) in keys_to_compact.items():
            # Re-read the full record value from old file
            old_reader = open(self._data_path(old_fid), "rb")
            old_reader.seek(old_offset)
            record_data = old_reader.read(old_size)
            old_reader.close()
            _, _, key_size, val_size = struct.unpack(HEADER_FORMAT, record_data[:HEADER_SIZE])
            value = record_data[HEADER_SIZE + key_size:].decode("utf-8")

            # Check if we need to rotate merged file
            if merged_file.tell() >= self.max_file_size:
                merged_file.close()
                self._write_hint_file(merged_file_id, hint_entries)
                self.file_handles[merged_file_id] = open(self._data_path(merged_file_id), "rb")
                hint_entries = []
                merged_file_id += 1
                merged_file = open(self._data_path(merged_file_id), "ab")

            key_bytes = key.encode("utf-8")
            val_bytes = value.encode("utf-8")
            payload = key_bytes + val_bytes
            new_crc = zlib.crc32(payload) & 0xFFFFFFFF
            header = struct.pack(HEADER_FORMAT, new_crc, ts, len(key_bytes), len(val_bytes))
            record = header + key_bytes + val_bytes
            new_offset = merged_file.tell()
            merged_file.write(record)

            new_entry = KeyEntry(merged_file_id, new_offset, len(record), ts)
            self.keydir[key] = new_entry
            hint_entries.append((key, new_entry))

        merged_file.flush()
        os.fsync(merged_file.fileno())
        merged_file.close()
        self._write_hint_file(merged_file_id, hint_entries)
        self.file_handles[merged_file_id] = open(self._data_path(merged_file_id), "rb")

        # Rename active file first (atomic on POSIX), then delete old files.
        # This ordering ensures a crash always leaves valid data on disk.
        self.active_file.close()
        old_active_id = self.active_file_id
        self.active_file_id = merged_file_id + 1
        os.rename(self._data_path(old_active_id),
                  self._data_path(self.active_file_id))
        dir_fd = os.open(self.data_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        for key, entry in self.keydir.items():
            if entry.file_id == old_active_id:
                entry.file_id = self.active_file_id
        if old_active_id in self.file_handles:
            self.file_handles[old_active_id].close()
            del self.file_handles[old_active_id]
        self._open_active_file()

        # Delete old immutable files (safe: merged file is already in place)
        for fid in immutable_ids:
            data_path = self._data_path(fid)
            hint_path = self._hint_path(fid)
            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(hint_path):
                os.remove(hint_path)

    def close(self) -> None:
        """Flush and close all file handles."""
        self.active_file.close()
        for fh in self.file_handles.values():
            fh.close()
        self.file_handles.clear()

    def __len__(self) -> int:
        return len(self.keydir)

    def __contains__(self, key: str) -> bool:
        return key in self.keydir
