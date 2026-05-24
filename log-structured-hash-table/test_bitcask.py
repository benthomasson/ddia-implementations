"""Tests for BitcaskStore."""

import os
import struct
import tempfile
import pytest
from bitcask import BitcaskStore, CorruptionError, HEADER_SIZE, HEADER_FMT


@pytest.fixture
def store_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(store_dir):
    s = BitcaskStore(store_dir, max_segment_size=100)
    yield s
    s.close()


# 1. Basic put/get
def test_basic_put_get(store):
    store.put("name", b"Alice")
    store.put("age", b"30")
    assert store.get("name") == b"Alice"
    assert store.get("age") == b"30"
    assert store.get("nonexistent") is None


# 2. Update overwrites
def test_update(store):
    store.put("name", b"Alice")
    store.put("name", b"Bob")
    assert store.get("name") == b"Bob"


# 3. Delete
def test_delete(store):
    store.put("key", b"value")
    assert store.delete("key") is True
    assert store.get("key") is None
    assert store.delete("key") is False  # already deleted


# 4. Segment rotation
def test_segment_rotation(store):
    # With max_segment_size=100, writing enough data should create multiple segments
    for i in range(20):
        store.put(f"key{i}", b"x" * 10)
    assert store.num_segments >= 2


# 5. Compaction preserves live data
def test_compaction(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)
    # Write and overwrite to create stale entries
    store.put("name", b"Alice")
    store.put("name", b"Bob")
    store.put("age", b"30")
    store.delete("age")
    # Force multiple segments
    for i in range(20):
        store.put(f"key{i}", b"x" * 10)

    assert store.num_segments >= 2
    stale = store.compact()
    assert stale >= 2  # old "Alice" and tombstone for "age"
    assert store.get("name") == b"Bob"
    assert store.get("age") is None
    for i in range(20):
        assert store.get(f"key{i}") == b"x" * 10
    store.close()


# 6. Disk usage decreases after compaction
def test_disk_usage_decreases(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)
    for i in range(10):
        store.put("key", f"value_{i}".encode())  # 10 overwrites = 9 stale
    for i in range(10):
        store.put(f"other{i}", b"data")
    usage_before = store.total_disk_usage
    store.compact()
    usage_after = store.total_disk_usage
    assert usage_after < usage_before
    store.close()


# 7. Crash recovery
def test_crash_recovery(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)
    store.put("name", b"Alice")
    store.put("city", b"NYC")
    for i in range(20):
        store.put(f"k{i}", b"v")
    store.close()

    store2 = BitcaskStore(store_dir, max_segment_size=100)
    assert store2.get("name") == b"Alice"
    assert store2.get("city") == b"NYC"
    assert len(store2) >= 22
    store2.close()


# 8. Hint files
def test_hint_files(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)
    store.put("name", b"Bob")
    for i in range(20):
        store.put(f"k{i}", b"v")
    store.compact()
    store.create_hint_files()
    store.close()

    store2 = BitcaskStore(store_dir, max_segment_size=100)
    assert store2.get("name") == b"Bob"
    store2.close()


# 9. CRC integrity
def test_crc_corruption(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=1024 * 1024, auto_compact_threshold=100)
    store.put("key", b"value")
    seg_path, offset = store._index["key"]

    # Corrupt a byte in the payload area while store is still open (index intact)
    with open(seg_path, "r+b") as f:
        f.seek(offset + HEADER_SIZE + 2)  # inside the key/value payload
        f.write(b"\xff")

    with pytest.raises(CorruptionError):
        store.get("key")
    store.close()


# 10. Iteration
def test_iteration(store):
    store.put("a", b"1")
    store.put("b", b"2")
    store.put("c", b"3")
    store.delete("b")
    items = dict(store)
    assert items == {"a": b"1", "c": b"3"}


# 11. Large dataset
def test_large_dataset(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=4096, auto_compact_threshold=100)
    n = 10000
    for i in range(n):
        store.put(f"key{i}", f"value{i}".encode())
    assert len(store) == n
    # Overwrite half
    for i in range(0, n, 2):
        store.put(f"key{i}", f"updated{i}".encode())
    store.compact()
    assert len(store) == n
    assert store.get("key0") == b"updated0"
    assert store.get("key1") == b"value1"
    store.close()


# 12. Contains
def test_contains(store):
    store.put("exists", b"yes")
    assert store.contains("exists") is True
    assert store.contains("nope") is False


# 13. Auto-compaction
def test_auto_compaction(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=50, auto_compact_threshold=3)
    # Write enough to trigger rotation beyond threshold
    for i in range(50):
        store.put(f"key{i}", b"x" * 5)
    # Auto-compact should have fired, reducing segment count
    assert store.num_segments <= 5  # should be compacted down
    # Data should still be intact
    for i in range(50):
        assert store.get(f"key{i}") == b"x" * 5
    store.close()


# 14. Partial write recovery
def test_partial_write_recovery(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=1024 * 1024, auto_compact_threshold=100)
    store.put("good", b"data")
    seg_path = store._active_path
    store.close()

    # Append a partial record (incomplete header + partial payload)
    with open(seg_path, "ab") as f:
        # Write a valid header but truncated payload
        key = b"broken"
        value = b"this_is_truncated"
        payload = key + value
        import zlib
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FMT, crc, len(key), len(value))
        f.write(header)
        f.write(key)  # write key but not value - partial!

    store2 = BitcaskStore(store_dir, max_segment_size=1024 * 1024)
    assert store2.get("good") == b"data"
    assert store2.get("broken") is None  # partial record skipped
    store2.close()
