"""Write-Ahead Log (WAL) for crash recovery."""

import os
import struct
import zlib
import threading
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional

OP_PUT = 1
OP_DELETE = 2
OP_COMMIT = 3
OP_CHECKPOINT = 4
OP_BEGIN = 5
OP_NAMES = {OP_PUT: "PUT", OP_DELETE: "DELETE", OP_COMMIT: "COMMIT",
            OP_CHECKPOINT: "CHECKPOINT", OP_BEGIN: "BEGIN"}
OP_BYTES = {v: k for k, v in OP_NAMES.items()}


@dataclass
class WALRecord:
    """A single WAL record."""
    seq_num: int
    op_type: str
    key: str
    value: str
    checksum: int


def _encode_record(seq_num: int, op_type_byte: int, key: bytes, value: bytes) -> bytes:
    """Encode a WAL record to binary format."""
    crc_data = struct.pack("B", op_type_byte) + key + value
    crc = zlib.crc32(crc_data) & 0xFFFFFFFF
    record_length = 4 + 8 + 1 + 4 + len(key) + 4 + len(value)
    header = struct.pack("<IIQBi", record_length, crc, seq_num, op_type_byte, len(key))
    return header + key + struct.pack("<i", len(value)) + value


def _read_record(f) -> Optional[WALRecord]:
    """Read one record from file. Returns None on EOF/partial read, raises on CRC error."""
    length_data = f.read(4)
    if len(length_data) < 4:
        return None
    record_length = struct.unpack("<I", length_data)[0]
    record_data = f.read(record_length)
    if len(record_data) < record_length:
        return None
    crc, seq_num, op_type_byte, key_len = struct.unpack_from("<IQBi", record_data, 0)
    offset = 4 + 8 + 1 + 4
    key = record_data[offset:offset + key_len]
    offset += key_len
    val_len = struct.unpack_from("<i", record_data, offset)[0]
    offset += 4
    value = record_data[offset:offset + val_len]
    crc_data = struct.pack("B", op_type_byte) + key + value
    expected_crc = zlib.crc32(crc_data) & 0xFFFFFFFF
    if crc != expected_crc:
        raise ValueError(f"CRC mismatch at seq {seq_num}")
    return WALRecord(seq_num, OP_NAMES.get(op_type_byte, "UNKNOWN"),
                     key.decode("utf-8"), value.decode("utf-8"), crc)


class WriteAheadLog:
    """Write-ahead log with crash recovery, batching, rotation, and truncation."""

    def __init__(self, log_dir: str, sync_mode: str = "sync",
                 max_file_size: int = 10 * 1024 * 1024,
                 batch_sync_count: int = 100):
        self._dir = log_dir
        self._sync_mode = sync_mode
        self._max_file_size = max_file_size
        self._batch_sync_count = batch_sync_count
        self._seq_num = 0
        self._write_count = 0
        self._lock = threading.Lock()
        os.makedirs(log_dir, exist_ok=True)
        self._seq_num = self._recover_seq_num()
        self._fd = None
        self._current_file = None
        self._open_latest()

    def _wal_files(self) -> List[str]:
        """Return sorted list of WAL file paths."""
        files = sorted(f for f in os.listdir(self._dir) if f.endswith(".wal"))
        return [os.path.join(self._dir, f) for f in files]

    def _recover_seq_num(self) -> int:
        """Scan all WAL files to find the highest sequence number."""
        max_seq = 0
        for path in self._wal_files():
            with open(path, "rb") as f:
                while True:
                    try:
                        rec = _read_record(f)
                        if rec is None:
                            break
                        max_seq = max(max_seq, rec.seq_num)
                    except ValueError:
                        break
        return max_seq

    def _open_latest(self):
        """Open the latest WAL file for appending, or create a new one."""
        files = self._wal_files()
        if files:
            last = files[-1]
            if os.path.getsize(last) < self._max_file_size:
                self._current_file = last
                self._fd = open(last, "ab")
                return
        self._rotate()

    def _rotate(self):
        """Create a new WAL file."""
        if self._fd:
            self._fd.flush()
            os.fsync(self._fd.fileno())
            self._fd.close()
        files = self._wal_files()
        next_num = 1
        if files:
            next_num = int(os.path.basename(files[-1]).split(".")[0]) + 1
        self._current_file = os.path.join(self._dir, f"{next_num:06d}.wal")
        self._fd = open(self._current_file, "ab")
        dir_fd = os.open(self._dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _do_sync(self, force: bool = False):
        """Fsync based on sync mode."""
        if self._sync_mode == "sync" or force:
            self._fd.flush()
            os.fsync(self._fd.fileno())
        elif self._sync_mode == "batch":
            self._write_count += 1
            if self._write_count >= self._batch_sync_count:
                self._fd.flush()
                os.fsync(self._fd.fileno())
                self._write_count = 0

    def _maybe_rotate(self):
        """Rotate if current file exceeds size limit."""
        if self._fd and self._fd.tell() >= self._max_file_size:
            self._rotate()

    def append(self, op_type: str, key: str, value: str = "") -> int:
        """Append a single record. Returns the sequence number."""
        with self._lock:
            self._seq_num += 1
            seq = self._seq_num
            data = _encode_record(seq, OP_BYTES[op_type],
                                  key.encode("utf-8"), value.encode("utf-8"))
            self._fd.write(data)
            self._do_sync()
            self._maybe_rotate()
            return seq

    def append_batch(self, operations: List[Tuple[str, str, str]]) -> int:
        """Atomically append a batch with BEGIN/COMMIT. Returns COMMIT sequence number."""
        with self._lock:
            buf = bytearray()
            self._seq_num += 1
            buf.extend(_encode_record(self._seq_num, OP_BEGIN, b"", b""))
            for op_type, key, value in operations:
                self._seq_num += 1
                buf.extend(_encode_record(self._seq_num, OP_BYTES[op_type],
                                          key.encode("utf-8"), value.encode("utf-8")))
            self._seq_num += 1
            commit_seq = self._seq_num
            buf.extend(_encode_record(commit_seq, OP_COMMIT, b"", b""))
            self._fd.write(bytes(buf))
            self._do_sync(force=True)
            self._maybe_rotate()
            return commit_seq

    def checkpoint(self) -> int:
        """Write a checkpoint record. Returns its sequence number."""
        with self._lock:
            self._seq_num += 1
            seq = self._seq_num
            self._fd.write(_encode_record(seq, OP_CHECKPOINT, b"", b""))
            self._do_sync(force=True)
            self._maybe_rotate()
            return seq

    def truncate(self, up_to_seq: int) -> None:
        """Remove all records with seq_num <= up_to_seq."""
        with self._lock:
            if self._fd:
                self._fd.flush()
                os.fsync(self._fd.fileno())
                self._fd.close()
                self._fd = None

            for path in self._wal_files():
                kept = []
                with open(path, "rb") as f:
                    while True:
                        try:
                            rec = _read_record(f)
                            if rec is None:
                                break
                            if rec.seq_num > up_to_seq:
                                kept.append(rec)
                        except ValueError:
                            break
                if not kept:
                    os.remove(path)
                else:
                    with open(path, "wb") as f:
                        for rec in kept:
                            f.write(_encode_record(rec.seq_num, OP_BYTES[rec.op_type],
                                                   rec.key.encode("utf-8"),
                                                   rec.value.encode("utf-8")))
                        f.flush()
                        os.fsync(f.fileno())
            self._open_latest()

    def replay(self, after_seq: int = 0) -> List[WALRecord]:
        """Replay committed records with seq_num > after_seq.

        Individual PUT/DELETE records outside a batch are included directly.
        Batch records (between BEGIN and COMMIT) are only included if the
        COMMIT is present; incomplete batches are discarded.
        """
        with self._lock:
            if self._fd:
                self._fd.flush()
        result = []
        in_batch = False
        batch_buf: List[WALRecord] = []
        for rec in self._read_all_records():
            if rec.seq_num <= after_seq:
                continue
            if rec.op_type == "BEGIN":
                in_batch = True
                batch_buf = []
            elif rec.op_type == "COMMIT":
                if in_batch:
                    result.extend(batch_buf)
                in_batch = False
                batch_buf = []
            elif rec.op_type in ("PUT", "DELETE"):
                if in_batch:
                    batch_buf.append(rec)
                else:
                    result.append(rec)
        return result

    def _read_all_records(self) -> Iterator[WALRecord]:
        """Read all valid records from all WAL files in sequence order.

        Verifies that sequence numbers are monotonically increasing and
        stops at any out-of-order record (treats it as corruption).
        """
        last_seq = 0
        for path in self._wal_files():
            with open(path, "rb") as f:
                while True:
                    try:
                        rec = _read_record(f)
                        if rec is None:
                            break
                        if rec.seq_num <= last_seq:
                            return
                        last_seq = rec.seq_num
                        yield rec
                    except ValueError:
                        return

    def iterate(self) -> Iterator[WALRecord]:
        """Iterate over all raw records including uncommitted ones."""
        with self._lock:
            if self._fd:
                self._fd.flush()
        yield from self._read_all_records()

    def current_seq_num(self) -> int:
        """Return the current sequence number."""
        return self._seq_num

    def close(self) -> None:
        """Flush and close the WAL."""
        with self._lock:
            if self._fd:
                self._fd.flush()
                os.fsync(self._fd.fileno())
                self._fd.close()
                self._fd = None
