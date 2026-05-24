"""Tests for BitcaskStore."""

import os
import struct
import tempfile
import zlib
import pytest
from bitcask import BitcaskStore, CorruptionError, HEADER_SIZE, HEADER_FMT


@pytest.fixture
def store_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def store(store_dir):
    s = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)
    yield s
    s.close()


# 1. Basic put/get + update + delete (from the example usage in the spec)
def test_example_usage(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)

    # Basic put/get
    store.put("name", b"Alice")
    store.put("age", b"30")
    assert store.get("name") == b"Alice"
    assert store.get("age") == b"30"
    assert store.get("nonexistent") is None

    # Update overwrites
    store.put("name", b"Bob")
    assert store.get("name") == b"Bob"

    # Delete
    store.delete("age")
    assert store.get("age") is None
    assert not store.contains("age")

    # Write more to force segment rotation
    for i in range(20):
        store.put(f"pad{i}", b"x" * 10)

    # Segments
    assert store.num_segments >= 2

    # Compaction
    stale_removed = store.compact()
    assert stale_removed >= 2  # old "Alice" and tombstone for "age"
    assert store.get("name") == b"Bob"
    assert store.get("age") is None

    # Iteration (dict(store) doesn't work because keys() triggers mapping protocol
    # without __getitem__; use list of tuples instead)
    store.put("city", b"NYC")
    all_items = {k: v for k, v in store}
    assert "name" in all_items
    assert all_items["name"] == b"Bob"
    assert "city" in all_items
    assert all_items["city"] == b"NYC"

    # Crash recovery
    store.close()
    store2 = BitcaskStore(store_dir, max_segment_size=100)
    assert store2.get("name") == b"Bob"
    assert store2.get("city") == b"NYC"
    assert len(store2) >= 2
    store2.close()


# 2. Segment rotation creates multiple files
def test_segment_rotation(store):
    for i in range(20):
        store.put(f"key{i}", b"x" * 10)
    assert store.num_segments >= 2


# 3. Compaction reduces disk usage and preserves data
def test_compaction_disk_usage(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=100, auto_compact_threshold=100)
    for i in range(10):
        store.put("key", f"value_{i}".encode())
    for i in range(10):
        store.put(f"other{i}", b"data")
    usage_before = store.total_disk_usage
    store.compact()
    usage_after = store.total_disk_usage
    assert usage_after < usage_before
    # All live data preserved
    assert store.get("key") == b"value_9"
    for i in range(10):
        assert store.get(f"other{i}") == b"data"
    store.close()


# 4. CRC32 corruption detection
def test_crc_corruption(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=1024 * 1024, auto_compact_threshold=100)
    store.put("key", b"value")
    seg_path, offset = store._index["key"]

    # Corrupt a byte in the payload
    with open(seg_path, "r+b") as f:
        f.seek(offset + HEADER_SIZE + 2)
        f.write(b"\xff")

    with pytest.raises(CorruptionError):
        store.get("key")
    store.close()


# 5. Hint files enable fast recovery
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
    for i in range(20):
        assert store2.get(f"k{i}") == b"v"
    store2.close()


# 6. Partial write recovery (simulated crash mid-write)
def test_partial_write_recovery(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=1024 * 1024, auto_compact_threshold=100)
    store.put("good", b"data")
    seg_path = store._active_path
    store.close()

    # Append a partial record
    with open(seg_path, "ab") as f:
        key = b"broken"
        value = b"this_is_truncated"
        payload = key + value
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(HEADER_FMT, crc, len(key), len(value))
        f.write(header)
        f.write(key)  # write key but not value

    store2 = BitcaskStore(store_dir, max_segment_size=1024 * 1024)
    assert store2.get("good") == b"data"
    assert store2.get("broken") is None
    store2.close()


# 7. Auto-compaction triggers when threshold exceeded
def test_auto_compaction(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=50, auto_compact_threshold=3)
    for i in range(50):
        store.put(f"key{i}", b"x" * 5)
    assert store.num_segments <= 5
    for i in range(50):
        assert store.get(f"key{i}") == b"x" * 5
    store.close()


# 8. Large dataset correctness
def test_large_dataset(store_dir):
    store = BitcaskStore(store_dir, max_segment_size=4096, auto_compact_threshold=100)
    n = 10000
    for i in range(n):
        store.put(f"key{i}", f"value{i}".encode())
    assert len(store) == n
    for i in range(0, n, 2):
        store.put(f"key{i}", f"updated{i}".encode())
    store.compact()
    assert len(store) == n
    assert store.get("key0") == b"updated0"
    assert store.get("key1") == b"value1"
    store.close()
