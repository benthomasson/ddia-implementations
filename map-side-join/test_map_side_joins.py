"""Tests for map-side join implementations."""

import pytest
from map_side_joins import (
    BroadcastHashJoin,
    PartitionedHashJoin,
    SortMergeJoin,
    JoinResult,
    partition_dataset,
    sort_dataset,
    compare_join_strategies,
)


# --- Fixtures ---

@pytest.fixture
def users():
    return [
        {"user_id": 1, "name": "Alice", "city": "NYC"},
        {"user_id": 2, "name": "Bob", "city": "SF"},
        {"user_id": 3, "name": "Carol", "city": "NYC"},
        {"user_id": 4, "name": "Dave", "city": "LA"},
    ]


@pytest.fixture
def orders():
    return [
        {"order_id": 101, "user_id": 1, "product": "Widget", "amount": 25.0},
        {"order_id": 102, "user_id": 1, "product": "Gadget", "amount": 50.0},
        {"order_id": 103, "user_id": 2, "product": "Widget", "amount": 25.0},
        {"order_id": 104, "user_id": 3, "product": "Doohickey", "amount": 15.0},
        {"order_id": 105, "user_id": 5, "product": "Thingamajig", "amount": 30.0},
    ]


# --- 1. Inner join with all three strategies ---

def test_inner_join_all_strategies(users, orders):
    bhj = BroadcastHashJoin(users, small_key="user_id", num_mappers=2)
    b_result = bhj.join(orders, join_type="inner")

    phj = PartitionedHashJoin(num_partitions=3, left_key="user_id")
    p_result = phj.join(users, orders, join_type="inner")

    smj = SortMergeJoin(left_key="user_id")
    s_result = smj.join(users, orders, join_type="inner")

    assert b_result.count == 4
    assert p_result.count == 4
    assert s_result.count == 4

    def order_ids(result):
        return sorted(r["order_id"] for r in result.records)

    assert order_ids(b_result) == [101, 102, 103, 104]
    assert order_ids(p_result) == [101, 102, 103, 104]
    assert order_ids(s_result) == [101, 102, 103, 104]


# --- 2. Left join ---

def test_left_join_broadcast(users, orders):
    bhj = BroadcastHashJoin(users, small_key="user_id", num_mappers=2)
    result = bhj.join(orders, join_type="left")
    assert result.count == 5
    unmatched = [r for r in result.records if r.get("name") is None]
    assert len(unmatched) == 1
    assert unmatched[0]["user_id"] == 5


def test_left_join_partitioned(users, orders):
    phj = PartitionedHashJoin(num_partitions=3, left_key="user_id")
    # For partitioned hash join, left=users, right=orders
    # Left join keeps all left records (users) even without matching orders
    result = phj.join(users, orders, join_type="left")
    # user_id 4 has no orders
    unmatched = [r for r in result.records if r.get("order_id") is None]
    assert len(unmatched) == 1
    assert unmatched[0]["user_id"] == 4


def test_left_join_sort_merge(users, orders):
    smj = SortMergeJoin(left_key="user_id")
    result = smj.join(users, orders, join_type="left")
    # Left join keeps all left records (users)
    assert result.count == 5  # 4 matched + 1 unmatched (Dave)
    unmatched = [r for r in result.records if r.get("order_id") is None]
    assert len(unmatched) == 1
    assert unmatched[0]["user_id"] == 4


# --- 3. One-to-many join ---

def test_one_to_many(users, orders):
    bhj = BroadcastHashJoin(users, small_key="user_id", num_mappers=2)
    result = bhj.join(orders, join_type="inner")
    alice_orders = [r for r in result.records if r["name"] == "Alice"]
    assert len(alice_orders) == 2


# --- 4. Many-to-many join ---

def test_many_to_many():
    left = [
        {"key": 1, "left_val": "a"},
        {"key": 1, "left_val": "b"},
    ]
    right = [
        {"key": 1, "right_val": "x"},
        {"key": 1, "right_val": "y"},
        {"key": 1, "right_val": "z"},
    ]
    bhj = BroadcastHashJoin(left, small_key="key", num_mappers=1)
    result = bhj.join(right, join_type="inner")
    assert result.count == 6  # 2 * 3 Cartesian product

    smj = SortMergeJoin(left_key="key")
    s_result = smj.join(left, right, join_type="inner")
    assert s_result.count == 6


# --- 5. No-match case ---

def test_no_match():
    left = [{"key": 1, "val": "a"}]
    right = [{"key": 2, "val": "b"}]
    bhj = BroadcastHashJoin(left, small_key="key", num_mappers=1)
    result = bhj.join(right, join_type="inner")
    assert result.count == 0


# --- 6. Empty datasets ---

def test_empty_datasets():
    bhj = BroadcastHashJoin([], small_key="key", num_mappers=1)
    result = bhj.join([{"key": 1}], join_type="inner")
    assert result.count == 0

    result_left = bhj.join([{"key": 1}], join_type="left")
    assert result_left.count == 1

    bhj2 = BroadcastHashJoin([{"key": 1}], small_key="key", num_mappers=1)
    result2 = bhj2.join([], join_type="inner")
    assert result2.count == 0

    phj = PartitionedHashJoin(num_partitions=2, left_key="key")
    result3 = phj.join([], [], join_type="inner")
    assert result3.count == 0

    smj = SortMergeJoin(left_key="key")
    result4 = smj.join([], [], join_type="inner")
    assert result4.count == 0


# --- 7. Field name conflict resolution ---

def test_field_name_conflict():
    left = [{"key": 1, "value": "left_val", "unique_left": "L"}]
    right = [{"key": 1, "value": "right_val", "unique_right": "R"}]

    bhj = BroadcastHashJoin(left, small_key="key", num_mappers=1)
    result = bhj.join(right, join_type="inner")
    assert result.count == 1
    rec = result.records[0]
    assert rec["key"] == 1
    assert rec["left_value"] == "left_val"
    assert rec["right_value"] == "right_val"
    assert rec["unique_left"] == "L"
    assert rec["unique_right"] == "R"


# --- 8. Partitioned hash join partitioning ---

def test_partitioned_join_independence(users, orders):
    phj = PartitionedHashJoin(num_partitions=4, left_key="user_id")
    result = phj.join(users, orders, join_type="inner")
    assert result.count == 4
    # Verify mapper_id is set (one per partition)
    mapper_ids = {r["_mapper_id"] for r in result.records}
    assert all(0 <= m < 4 for m in mapper_ids)


# --- 9. Sort-merge with sorted and unsorted input ---

def test_sort_merge_presorted():
    left = [{"key": 1, "v": "a"}, {"key": 2, "v": "b"}]
    right = [{"key": 1, "v": "x"}, {"key": 2, "v": "y"}]
    smj = SortMergeJoin(left_key="key")
    result = smj.join(left, right, join_type="inner")
    assert result.count == 2
    assert result.stats["sorted_input"] is True


def test_sort_merge_unsorted():
    left = [{"key": 2, "v": "b"}, {"key": 1, "v": "a"}]
    right = [{"key": 2, "v": "y"}, {"key": 1, "v": "x"}]
    smj = SortMergeJoin(left_key="key")
    result = smj.join(left, right, join_type="inner")
    assert result.count == 2
    assert result.stats["sorted_input"] is False


# --- 10. Compare join strategies ---

def test_compare_strategies(users, orders):
    comparison = compare_join_strategies(users, orders, "user_id")
    assert comparison["verification"] is True
    assert "broadcast" in comparison
    assert "partitioned" in comparison
    assert "sort_merge" in comparison


# --- 11. partition_dataset utility ---

def test_partition_dataset(orders):
    partitions = partition_dataset(orders, "user_id", 3)
    assert len(partitions) == 3
    assert sum(len(p) for p in partitions) == 5


# --- 12. Performance stats ---

def test_performance_stats(users, orders):
    bhj = BroadcastHashJoin(users, small_key="user_id", num_mappers=2)
    result = bhj.join(orders, join_type="inner")
    stats = result.stats
    assert stats["records_read_left"] == 4
    assert stats["records_read_right"] == 5
    assert stats["output_records"] == 4
    assert stats["mappers_used"] == 2
    assert stats["hash_table_size"] == 4
    assert "hash_lookups" in stats


# --- 13. Large dataset ---

def test_large_dataset():
    import random
    random.seed(42)
    left = [{"key": i, "left_val": f"L{i}"} for i in range(1000)]
    right = [{"key": random.randint(0, 999), "right_val": f"R{i}"} for i in range(2000)]

    bhj = BroadcastHashJoin(left, small_key="key", num_mappers=4)
    b_result = bhj.join(right, join_type="inner")

    phj = PartitionedHashJoin(num_partitions=8, left_key="key")
    p_result = phj.join(left, right, join_type="inner")

    smj = SortMergeJoin(left_key="key")
    s_result = smj.join(left, right, join_type="inner")

    # All should produce same count
    assert b_result.count == p_result.count == s_result.count
    assert b_result.count == 2000  # All right records have key in [0,999]


# --- 14. Broadcast with different mapper counts ---

def test_broadcast_mapper_counts(users, orders):
    results = []
    for n in [1, 2, 4, 8]:
        bhj = BroadcastHashJoin(users, small_key="user_id", num_mappers=n)
        res = bhj.join(orders, join_type="inner")
        results.append(sorted(rec["order_id"] for rec in res.records))

    assert all(r == results[0] for r in results)


# --- 15. Missing join key ---

def test_missing_join_key():
    left = [{"key": 1, "v": "a"}, {"v": "no_key"}, {"key": 3, "v": "c"}]
    right = [{"key": 1, "v": "x"}, {"key": 3, "v": "z"}]

    bhj = BroadcastHashJoin(left, small_key="key", num_mappers=1)
    result = bhj.join(right, join_type="inner")
    assert result.count == 2
    assert result.stats["skipped_records"] == 1

    smj = SortMergeJoin(left_key="key")
    s_result = smj.join(left, right, join_type="inner")
    assert s_result.count == 2
    assert s_result.stats["skipped_records"] == 1
