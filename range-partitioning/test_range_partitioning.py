"""Tests for range-based partitioning."""

import pytest
from range_partitioning import Partition, RangePartitionedStore


# 1. Basic put/get/delete with a single partition
def test_basic_put_get_delete():
    store = RangePartitionedStore(max_partition_size=100)
    store.put("apple", 1)
    store.put("banana", 2)
    store.put("cherry", 3)

    assert store.get("apple") == 1
    assert store.get("banana") == 2
    assert store.get("cherry") == 3
    assert store.get("missing") is None

    assert store.delete("banana") is True
    assert store.get("banana") is None
    assert store.delete("banana") is False
    assert store.partition_count == 1


# 2. Auto-split triggers when partition exceeds max_partition_size
def test_auto_split_triggers():
    store = RangePartitionedStore(max_partition_size=4)
    for c in "abcde":
        store.put(c, ord(c))
    assert store.partition_count == 2


# 3. After split, both partitions have roughly equal sizes
def test_split_produces_equal_partitions():
    store = RangePartitionedStore(max_partition_size=10)
    for i in range(11):
        store.put(f"key_{i:03d}", i)
    assert store.partition_count == 2
    info = store.get_partition_info()
    assert abs(info[0].size - info[1].size) <= 1


# 4. Partition boundaries are contiguous
def test_boundaries_contiguous():
    store = RangePartitionedStore(max_partition_size=5)
    for i in range(30):
        store.put(f"k{i:04d}", i)
    info = store.get_partition_info()
    assert info[0].start_key == ""
    assert info[-1].end_key is None
    for i in range(len(info) - 1):
        assert info[i].end_key == info[i + 1].start_key


# 5. Range scan within a single partition
def test_range_scan_single_partition():
    store = RangePartitionedStore(max_partition_size=100)
    store.put("a", 1)
    store.put("b", 2)
    store.put("c", 3)
    store.put("d", 4)
    result = store.range_scan("b", "d")
    assert result == [("b", 2), ("c", 3)]


# 6. Range scan spanning multiple partitions
def test_range_scan_multiple_partitions():
    store = RangePartitionedStore(max_partition_size=4)
    for c in "abcdefgh":
        store.put(c, ord(c))
    assert store.partition_count >= 2
    result = store.range_scan("b", "g")
    keys = [k for k, v in result]
    assert keys == ["b", "c", "d", "e", "f"]


# 7. Range scan with no end_key
def test_range_scan_no_end():
    store = RangePartitionedStore(max_partition_size=4)
    for c in "abcdefgh":
        store.put(c, ord(c))
    result = store.range_scan("e")
    keys = [k for k, v in result]
    assert keys == ["e", "f", "g", "h"]


# 8. merge_small_partitions merges adjacent small partitions
def test_merge_small_partitions():
    store = RangePartitionedStore(max_partition_size=4, min_partition_size=3)
    for c in "abcde":
        store.put(c, ord(c))
    assert store.partition_count >= 2
    # Delete enough to make partitions small
    store.delete("c")
    store.delete("d")
    store.delete("e")
    merged = store.merge_small_partitions()
    assert merged >= 1
    assert store.partition_count == 1
    # Data still intact
    assert store.get("a") == ord("a")
    assert store.get("b") == ord("b")


# 9. Merge does not merge if combined size exceeds min_partition_size
def test_merge_respects_threshold():
    store = RangePartitionedStore(max_partition_size=4, min_partition_size=3)
    # Insert enough to split
    for c in "abcde":
        store.put(c, ord(c))
    count_before = store.partition_count
    # Both partitions have enough keys combined >= min_partition_size
    merged = store.merge_small_partitions()
    # Should not merge since combined >= 3
    assert merged == 0


# 10. get_partition_for_key returns correct partition after splits
def test_routing_after_splits():
    store = RangePartitionedStore(max_partition_size=4)
    keys = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    for k in keys:
        store.put(k, k.upper())
    for k in keys:
        info = store.get_partition_for_key(k)
        assert info.start_key <= k
        assert info.end_key is None or k < info.end_key
        assert store.get(k) == k.upper()


# 11. Large number of keys produces balanced partitions
def test_large_scale_balanced():
    store = RangePartitionedStore(max_partition_size=100, min_partition_size=20)
    for i in range(10_000):
        store.put(f"key_{i:06d}", i)
    assert store.total_keys == 10_000
    info = store.get_partition_info()
    sizes = [p.size for p in info]
    # No partition should be more than max_partition_size
    assert all(s <= 100 for s in sizes)
    # Boundaries contiguous
    assert info[0].start_key == ""
    assert info[-1].end_key is None
    for i in range(len(info) - 1):
        assert info[i].end_key == info[i + 1].start_key


# 12. Delete followed by merge
def test_delete_then_merge():
    store = RangePartitionedStore(max_partition_size=10, min_partition_size=8)
    for i in range(25):
        store.put(f"k{i:03d}", i)
    initial_partitions = store.partition_count
    assert initial_partitions > 1
    # Delete many keys
    for i in range(20):
        store.delete(f"k{i:03d}")
    assert store.total_keys == 5
    merged = store.merge_small_partitions()
    assert merged >= 1
    assert store.partition_count < initial_partitions
    # Remaining data intact
    for i in range(20, 25):
        assert store.get(f"k{i:03d}") == i


# Test the example usage from the spec
def test_example_usage():
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


# Test update (put same key twice)
def test_update_value():
    store = RangePartitionedStore()
    store.put("key", "old")
    store.put("key", "new")
    assert store.get("key") == "new"
    assert store.total_keys == 1
