"""Tests for range-based partitioning with dynamic split and merge."""


import pytest
from range_partitioning import Partition, RangePartitionedStore, PartitionInfo


def test_example_usage():
    """Test the exact example from the spec."""
    store = RangePartitionedStore(max_partition_size=4, min_partition_size=2)
    assert store.partition_count == 1

    store.put("banana", 2)
    store.put("apple", 1)
    store.put("cherry", 3)
    store.put("date", 4)

    assert store.get("apple") == 1
    assert store.get("banana") == 2

    store.put("elderberry", 5)
    assert store.partition_count == 2

    results = store.range_scan("banana", "elderberry")
    assert results == [("banana", 2), ("cherry", 3), ("date", 4)]

    all_items = store.range_scan("")
    assert [k for k, v in all_items] == ["apple", "banana", "cherry", "date", "elderberry"]

    info = store.get_partition_info()
    assert len(info) == 2
    assert info[0].end_key == info[1].start_key

    store.delete("cherry")
    store.delete("date")
    store.delete("elderberry")
    merged = store.merge_small_partitions()
    assert merged >= 1
    assert store.partition_count == 1


def test_basic_put_get_delete():
    """Basic CRUD operations on a single partition."""
    store = RangePartitionedStore(max_partition_size=100)
    store.put("apple", 1)
    store.put("banana", 2)

    assert store.get("apple") == 1
    assert store.get("banana") == 2
    assert store.get("missing") is None

    assert store.delete("apple") is True
    assert store.get("apple") is None
    assert store.delete("apple") is False

    # Update existing key
    store.put("banana", 99)
    assert store.get("banana") == 99
    assert store.total_keys == 1


def test_auto_split_and_balance():
    """Split triggers at threshold and produces balanced partitions."""
    store = RangePartitionedStore(max_partition_size=10)
    for i in range(11):
        store.put(f"key_{i:03d}", i)
    assert store.partition_count == 2
    info = store.get_partition_info()
    assert abs(info[0].size - info[1].size) <= 1


def test_contiguous_boundaries():
    """Partition boundaries have no gaps or overlaps after multiple splits."""
    store = RangePartitionedStore(max_partition_size=5)
    for i in range(30):
        store.put(f"k{i:04d}", i)
    info = store.get_partition_info()
    assert info[0].start_key == ""
    assert info[-1].end_key is None
    for i in range(len(info) - 1):
        assert info[i].end_key == info[i + 1].start_key


def test_range_scan_across_partitions():
    """Range scan spanning multiple partitions returns sorted results."""
    store = RangePartitionedStore(max_partition_size=4)
    for c in "abcdefgh":
        store.put(c, ord(c))
    assert store.partition_count >= 2

    result = store.range_scan("b", "g")
    keys = [k for k, v in result]
    assert keys == ["b", "c", "d", "e", "f"]

    # No end_key - scan to end
    result2 = store.range_scan("e")
    keys2 = [k for k, v in result2]
    assert keys2 == ["e", "f", "g", "h"]


def test_merge_respects_max_size():
    """Merge should not combine partitions whose combined size > max."""
    store = RangePartitionedStore(max_partition_size=4, min_partition_size=3)
    for c in "abcde":
        store.put(c, ord(c))
    # After split, partitions should not merge since combined > min
    count_before = store.partition_count
    merged = store.merge_small_partitions()
    assert merged == 0


def test_routing_after_splits():
    """get_partition_for_key routes correctly after multiple splits."""
    store = RangePartitionedStore(max_partition_size=5)
    keys = [f"key_{i:03d}" for i in range(25)]
    for k in keys:
        store.put(k, k)
    for k in keys:
        info = store.get_partition_for_key(k)
        assert info.start_key <= k
        assert info.end_key is None or k < info.end_key
        assert store.get(k) == k


def test_large_dataset():
    """10K+ keys produce balanced partitions with all data intact."""
    store = RangePartitionedStore(max_partition_size=100, min_partition_size=20)
    n = 10000
    for i in range(n):
        store.put(f"k{i:06d}", i)
    assert store.total_keys == n
    assert store.partition_count > 1

    # Verify contiguous boundaries
    info = store.get_partition_info()
    assert info[0].start_key == ""
    assert info[-1].end_key is None
    for i in range(len(info) - 1):
        assert info[i].end_key == info[i + 1].start_key

    # Spot check a few values
    assert store.get("k000000") == 0
    assert store.get("k005000") == 5000
    assert store.get("k009999") == 9999


def test_delete_then_merge():
    """Delete enough keys to trigger merges."""
    store = RangePartitionedStore(max_partition_size=4, min_partition_size=4)
    for c in "abcdefghij":
        store.put(c, ord(c))
    assert store.partition_count >= 2

    # Delete most keys
    for c in "bcdefghi":
        store.delete(c)
    assert store.total_keys == 2

    merged = store.merge_small_partitions()
    assert merged >= 1
    # All remaining data still accessible
    assert store.get("a") == ord("a")
    assert store.get("j") == ord("j")


def test_unique_partition_ids():
    """All partition IDs must be unique."""
    store = RangePartitionedStore(max_partition_size=3)
    for i in range(20):
        store.put(f"key_{i:03d}", i)
    info = store.get_partition_info()
    ids = [p.partition_id for p in info]
    assert len(ids) == len(set(ids))
