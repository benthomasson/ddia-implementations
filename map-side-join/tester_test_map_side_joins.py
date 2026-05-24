"""Tests for map-side join implementations."""

import sys

from map_side_joins import (
    BroadcastHashJoin, PartitionedHashJoin, SortMergeJoin,
    partition_dataset, sort_dataset, compare_join_strategies,
)

# --- Shared test data ---

USERS = [
    {"user_id": 1, "name": "Alice", "city": "NYC"},
    {"user_id": 2, "name": "Bob", "city": "SF"},
    {"user_id": 3, "name": "Carol", "city": "NYC"},
    {"user_id": 4, "name": "Dave", "city": "LA"},
]

ORDERS = [
    {"order_id": 101, "user_id": 1, "product": "Widget", "amount": 25.0},
    {"order_id": 102, "user_id": 1, "product": "Gadget", "amount": 50.0},
    {"order_id": 103, "user_id": 2, "product": "Widget", "amount": 25.0},
    {"order_id": 104, "user_id": 3, "product": "Doohickey", "amount": 15.0},
    {"order_id": 105, "user_id": 5, "product": "Thingamajig", "amount": 30.0},
]

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
            errors.append((name, str(e)))
    return decorator


# --- Test 1: Inner join with all three strategies (spec example) ---

@test("inner join - all strategies produce same results")
def _():
    bhj = BroadcastHashJoin(USERS, small_key="user_id", num_mappers=2)
    b = bhj.join(ORDERS, join_type="inner")

    phj = PartitionedHashJoin(num_partitions=3, left_key="user_id")
    p = phj.join(USERS, ORDERS, join_type="inner")

    smj = SortMergeJoin(left_key="user_id")
    s = smj.join(USERS, ORDERS, join_type="inner")

    assert b.count == 4, f"broadcast count={b.count}"
    assert p.count == 4, f"partitioned count={p.count}"
    assert s.count == 4, f"sort-merge count={s.count}"

    expected_ids = {101, 102, 103, 104}
    assert set(r["order_id"] for r in b.records) == expected_ids
    assert set(r["order_id"] for r in p.records) == expected_ids
    assert set(r["order_id"] for r in s.records) == expected_ids

    # All records have fields from both datasets
    for r in b.records:
        assert "name" in r
        assert "product" in r
        assert "user_id" in r


# --- Test 2: Left join keeps unmatched records ---

@test("left join - broadcast keeps unmatched large-side records")
def _():
    bhj = BroadcastHashJoin(USERS, small_key="user_id", num_mappers=2)
    r = bhj.join(ORDERS, join_type="left")
    assert r.count == 5, f"count={r.count}"
    unmatched = [x for x in r.records if x.get("name") is None]
    assert len(unmatched) == 1
    assert unmatched[0]["user_id"] == 5


@test("left join - partitioned keeps unmatched left records")
def _():
    phj = PartitionedHashJoin(num_partitions=3, left_key="user_id")
    r = phj.join(USERS, ORDERS, join_type="left")
    # user_id=4 (Dave) has no orders
    unmatched = [x for x in r.records if x.get("order_id") is None]
    assert len(unmatched) == 1, f"unmatched count={len(unmatched)}"
    assert unmatched[0]["user_id"] == 4


@test("left join - sort-merge keeps unmatched left records")
def _():
    smj = SortMergeJoin(left_key="user_id")
    r = smj.join(USERS, ORDERS, join_type="left")
    assert r.count == 5, f"count={r.count}"
    unmatched = [x for x in r.records if x.get("order_id") is None]
    assert len(unmatched) == 1
    assert unmatched[0]["user_id"] == 4


# --- Test 3: One-to-many join ---

@test("one-to-many - Alice has 2 orders")
def _():
    bhj = BroadcastHashJoin(USERS, small_key="user_id", num_mappers=2)
    r = bhj.join(ORDERS, join_type="inner")
    alice = [x for x in r.records if x["name"] == "Alice"]
    assert len(alice) == 2, f"Alice records={len(alice)}"


# --- Test 4: Many-to-many (Cartesian product) ---

@test("many-to-many - Cartesian product of duplicate keys")
def _():
    left = [{"key": 1, "lv": "a"}, {"key": 1, "lv": "b"}]
    right = [{"key": 1, "rv": "x"}, {"key": 1, "rv": "y"}, {"key": 1, "rv": "z"}]

    bhj = BroadcastHashJoin(left, small_key="key", num_mappers=1)
    b = bhj.join(right, join_type="inner")
    assert b.count == 6, f"broadcast count={b.count}"

    smj = SortMergeJoin(left_key="key")
    s = smj.join(left, right, join_type="inner")
    assert s.count == 6, f"sort-merge count={s.count}"

    phj = PartitionedHashJoin(num_partitions=2, left_key="key")
    p = phj.join(left, right, join_type="inner")
    assert p.count == 6, f"partitioned count={p.count}"


# --- Test 5: Empty datasets ---

@test("empty datasets - inner join produces no results")
def _():
    bhj = BroadcastHashJoin([], small_key="k", num_mappers=1)
    assert bhj.join([{"k": 1}], join_type="inner").count == 0
    assert bhj.join([], join_type="inner").count == 0

    smj = SortMergeJoin(left_key="k")
    assert smj.join([], [{"k": 1}], join_type="inner").count == 0
    assert smj.join([{"k": 1}], [], join_type="inner").count == 0


# --- Test 6: Field name conflict resolution ---

@test("field name conflicts resolved with left_/right_ prefixes")
def _():
    left = [{"id": 1, "val": "left_val"}]
    right = [{"id": 1, "val": "right_val"}]

    bhj = BroadcastHashJoin(left, small_key="id", num_mappers=1)
    r = bhj.join(right, join_type="inner")
    assert r.count == 1
    rec = r.records[0]
    assert rec["id"] == 1
    assert rec["left_val"] == "left_val"
    assert rec["right_val"] == "right_val"
    assert "val" not in rec  # original field replaced by prefixed versions


# --- Test 7: compare_join_strategies ---

@test("compare_join_strategies verifies all produce same results")
def _():
    comparison = compare_join_strategies(USERS, ORDERS, "user_id")
    assert comparison["verification"] is True
    assert "broadcast" in comparison
    assert "partitioned" in comparison
    assert "sort_merge" in comparison


# --- Test 8: partition_dataset and sort_dataset utilities ---

@test("partition_dataset distributes all records")
def _():
    parts = partition_dataset(ORDERS, "user_id", 3)
    assert sum(len(p) for p in parts) == 5
    assert len(parts) == 3


@test("sort_dataset returns sorted copy")
def _():
    unsorted = [{"k": 3}, {"k": 1}, {"k": 2}]
    s = sort_dataset(unsorted, "k")
    assert [r["k"] for r in s] == [1, 2, 3]
    # Original unchanged
    assert unsorted[0]["k"] == 3


# --- Test 9: Performance stats ---

@test("stats contain expected keys")
def _():
    bhj = BroadcastHashJoin(USERS, small_key="user_id", num_mappers=2)
    r = bhj.join(ORDERS, join_type="inner")
    s = r.stats
    assert s["records_read_left"] == 4
    assert s["records_read_right"] == 5
    assert s["output_records"] == 4
    assert s["mappers_used"] == 2
    assert "hash_lookups" in s
    assert "hash_table_size" in s


# --- Test 10: Records missing join key are skipped ---

@test("records missing join key are skipped gracefully")
def _():
    left = [{"id": 1, "v": "a"}, {"v": "no_key"}]
    right = [{"id": 1, "w": "b"}, {"w": "no_key"}]

    bhj = BroadcastHashJoin(left, small_key="id", num_mappers=1)
    r = bhj.join(right, join_type="inner")
    assert r.count == 1
    assert r.stats["skipped_records"] == 2  # one from each side


# --- Summary ---

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for name, err in errors:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("All tests passed!")
