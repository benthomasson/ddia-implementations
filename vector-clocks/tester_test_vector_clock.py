"""Tests for vector clock implementation."""

import pytest
from vector_clock import VectorClock, VersionedValue, VersionedKVStore, find_conflicts


# 1. VectorClock comparison
class TestComparison:
    def test_before(self):
        vc1 = VectorClock({"A": 1, "B": 2})
        vc2 = VectorClock({"A": 1, "B": 3})
        assert vc1.compare(vc2) == "BEFORE"

    def test_after(self):
        vc1 = VectorClock({"A": 1, "B": 3})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.compare(vc2) == "AFTER"

    def test_equal(self):
        vc1 = VectorClock({"A": 1, "B": 2})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.compare(vc2) == "EQUAL"

    def test_concurrent(self):
        vc1 = VectorClock({"A": 2, "B": 1})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.compare(vc2) == "CONCURRENT"

    def test_is_concurrent(self):
        vc1 = VectorClock({"A": 2, "B": 1})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.is_concurrent(vc2) is True
        assert VectorClock({"A": 1}).is_concurrent(VectorClock({"A": 2})) is False


# 2. Increment
class TestIncrement:
    def test_increment_existing(self):
        vc = VectorClock({"A": 1, "B": 2})
        vc2 = vc.increment("A")
        assert vc2 == VectorClock({"A": 2, "B": 2})
        # Original unchanged
        assert vc == VectorClock({"A": 1, "B": 2})

    def test_increment_new_node(self):
        vc = VectorClock({"A": 1})
        vc2 = vc.increment("B")
        assert vc2 == VectorClock({"A": 1, "B": 1})

    def test_only_specified_node_changes(self):
        vc = VectorClock({"A": 1, "B": 2, "C": 3})
        vc2 = vc.increment("B")
        assert vc2.get("A") == 1
        assert vc2.get("B") == 3
        assert vc2.get("C") == 3


# 3. Merge
class TestMerge:
    def test_basic_merge(self):
        vc1 = VectorClock({"A": 1, "B": 3})
        vc2 = VectorClock({"A": 2, "B": 1})
        merged = vc1.merge(vc2)
        assert merged == VectorClock({"A": 2, "B": 3})

    def test_merge_disjoint(self):
        vc1 = VectorClock({"A": 1})
        vc2 = VectorClock({"B": 2})
        merged = vc1.merge(vc2)
        assert merged == VectorClock({"A": 1, "B": 2})

    def test_merge_with_empty(self):
        vc = VectorClock({"A": 1})
        merged = vc.merge(VectorClock())
        assert merged == vc


# 4. Dominates
class TestDominates:
    def test_equal_dominates(self):
        vc = VectorClock({"A": 1, "B": 2})
        assert vc.dominates(vc) is True

    def test_strictly_dominates(self):
        vc1 = VectorClock({"A": 2, "B": 3})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.dominates(vc2) is True
        assert vc2.dominates(vc1) is False

    def test_concurrent_no_domination(self):
        vc1 = VectorClock({"A": 2, "B": 1})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.dominates(vc2) is False
        assert vc2.dominates(vc1) is False

    def test_descends_from(self):
        vc1 = VectorClock({"A": 2, "B": 3})
        vc2 = VectorClock({"A": 1, "B": 2})
        assert vc1.descends_from(vc2) is True
        assert vc2.descends_from(vc1) is False


# 5. Versioned store put/get
class TestStorePutGet:
    def test_basic_write_read(self):
        store = VersionedKVStore("A")
        ctx = store.put("key", "value")
        versions = store.get("key")
        assert len(versions) == 1
        assert versions[0].value == "value"
        assert versions[0].vector_clock.get("A") == 1

    def test_get_nonexistent(self):
        store = VersionedKVStore("A")
        assert store.get("missing") == []

    def test_keys(self):
        store = VersionedKVStore("A")
        store.put("x", "1")
        store.put("y", "2")
        assert sorted(store.keys()) == ["x", "y"]


# 6. Conflict creation
class TestConflictCreation:
    def test_concurrent_writes_create_siblings(self):
        # Concurrent writes require different coordinator nodes
        store_a = VersionedKVStore("A")
        store_b = VersionedKVStore("B")
        # Client 1 writes to store A
        ctx_a = store_a.put("cart", "milk,eggs")
        # Client 2 writes to store B concurrently
        ctx_b = store_b.put("cart", "milk,bread")
        # Verify the clocks are concurrent
        assert ctx_a.is_concurrent(ctx_b)
        # Simulate anti-entropy: store_a receives store_b's version
        store_a._receive_replica("cart", "milk,bread", ctx_b)
        versions = store_a.get("cart")
        assert len(versions) == 2
        values = {v.value for v in versions}
        assert values == {"milk,eggs", "milk,bread"}

    def test_concurrent_with_shared_context(self):
        # Two nodes diverge from a shared initial version
        store_a = VersionedKVStore("A")
        store_b = VersionedKVStore("B")
        initial = store_a.put("cart", "milk")  # {A:1}
        # Both read the same version, write to different nodes
        ctx_a = store_a.put("cart", "milk,eggs", context=initial)   # {A:2}
        ctx_b = store_b.put("cart", "milk,bread", context=initial)  # {A:1, B:1}
        assert ctx_a.is_concurrent(ctx_b)
        # Merge into store_a
        store_a._receive_replica("cart", "milk,bread", ctx_b)
        versions = store_a.get("cart")
        assert len(versions) == 2

    def test_sequential_writes_no_siblings(self):
        store = VersionedKVStore("A")
        ctx1 = store.put("k", "v1")
        ctx2 = store.put("k", "v2", context=ctx1)
        versions = store.get("k")
        assert len(versions) == 1
        assert versions[0].value == "v2"


# 7. Conflict detection
class TestConflictDetection:
    def test_finds_conflicts(self):
        v1 = VersionedValue(value="a", vector_clock=VectorClock({"A": 2}))
        v2 = VersionedValue(value="b", vector_clock=VectorClock({"B": 1}))
        assert find_conflicts([v1, v2]) is True

    def test_no_conflicts_ordered(self):
        v1 = VersionedValue(value="a", vector_clock=VectorClock({"A": 1}))
        v2 = VersionedValue(value="b", vector_clock=VectorClock({"A": 2}))
        assert find_conflicts([v1, v2]) is False

    def test_no_conflicts_single(self):
        v1 = VersionedValue(value="a", vector_clock=VectorClock({"A": 1}))
        assert find_conflicts([v1]) is False

    def test_no_conflicts_empty(self):
        assert find_conflicts([]) is False


# 8. Reconciliation
class TestReconciliation:
    def test_reconcile_siblings(self):
        store_a = VersionedKVStore("A")
        store_b = VersionedKVStore("B")
        ctx_init = store_a.put("cart", "milk")
        ctx_a = store_a.put("cart", "milk,eggs", context=ctx_init)
        ctx_b = store_b.put("cart", "milk,bread", context=ctx_init)
        # Merge replicas into store_a
        store_a._receive_replica("cart", "milk,bread", ctx_b)
        siblings = store_a.get("cart")
        assert len(siblings) == 2

        merged_vc = store_a.reconcile("cart", "milk,eggs,bread",
                                       [s.vector_clock for s in siblings])
        versions = store_a.get("cart")
        assert len(versions) == 1
        assert versions[0].value == "milk,eggs,bread"
        # Merged clock dominates both siblings
        for s in siblings:
            assert merged_vc.dominates(s.vector_clock)

    def test_reconcile_history(self):
        store_a = VersionedKVStore("A")
        store_b = VersionedKVStore("B")
        ctx_init = store_a.put("k", "v1")
        store_a.put("k", "v2a", context=ctx_init)
        ctx_b = store_b.put("k", "v2b", context=ctx_init)
        store_a._receive_replica("k", "v2b", ctx_b)
        siblings = store_a.get("k")
        assert len(siblings) == 2
        store_a.reconcile("k", "merged", [s.vector_clock for s in siblings])
        reconcile_entries = [h for h in store_a.history if h.action == "reconciled"]
        assert len(reconcile_entries) == 1


# 9. Read-modify-write
class TestReadModifyWrite:
    def test_rmw_supersedes(self):
        store = VersionedKVStore("A")
        store.put("cart", "item1")
        versions = store.get("cart")
        assert len(versions) == 1
        store.put("cart", "item1,item2", context=versions[0].vector_clock)
        versions = store.get("cart")
        assert len(versions) == 1
        assert versions[0].value == "item1,item2"

    def test_rmw_chain(self):
        store = VersionedKVStore("A")
        ctx = store.put("k", "v1")
        ctx = store.put("k", "v2", context=ctx)
        ctx = store.put("k", "v3", context=ctx)
        versions = store.get("k")
        assert len(versions) == 1
        assert versions[0].value == "v3"


# 10. Edge cases
class TestEdgeCases:
    def test_empty_vector_clocks(self):
        vc1 = VectorClock()
        vc2 = VectorClock()
        assert vc1.compare(vc2) == "EQUAL"
        assert vc1.dominates(vc2) is True
        assert vc1.is_concurrent(vc2) is False

    def test_empty_vs_nonempty(self):
        vc1 = VectorClock()
        vc2 = VectorClock({"A": 1})
        assert vc1.compare(vc2) == "BEFORE"
        assert vc2.compare(vc1) == "AFTER"

    def test_single_node_system(self):
        store = VersionedKVStore("solo")
        ctx = store.put("k", "v1")
        ctx = store.put("k", "v2", context=ctx)
        versions = store.get("k")
        assert len(versions) == 1

    def test_many_concurrent_writers(self):
        # Simulate concurrent writes from different nodes sharing a store
        stores = [VersionedKVStore(f"node{i}") for i in range(5)]
        # All start from the same base version written by node0
        base_ctx = stores[0].put("k", "base")
        # Each node writes concurrently with the same base context
        # Collect all versions manually into one store
        all_versions = []
        for i, store in enumerate(stores):
            vc = store.put("k", f"v{i}", context=base_ctx)
            all_versions.append(VersionedValue(value=f"v{i}", vector_clock=vc))
        # All 5 versions should be concurrent with each other
        assert find_conflicts(all_versions) is True
        assert len(all_versions) == 5

    def test_reconcile_already_resolved(self):
        store = VersionedKVStore("A")
        ctx = store.put("k", "v1")
        # Only one version, reconcile should still work
        versions = store.get("k")
        new_vc = store.reconcile("k", "v1_resolved", [v.vector_clock for v in versions])
        versions = store.get("k")
        assert len(versions) == 1
        assert new_vc.dominates(ctx)

    def test_missing_node_ids_in_comparison(self):
        vc1 = VectorClock({"A": 1})
        vc2 = VectorClock({"B": 1})
        assert vc1.compare(vc2) == "CONCURRENT"

    def test_zero_entries_stripped(self):
        vc = VectorClock({"A": 0, "B": 1})
        assert vc.get("A") == 0
        assert vc == VectorClock({"B": 1})

    def test_prune(self):
        vc = VectorClock({"A": 5, "B": 3, "C": 1, "D": 4})
        pruned = vc.prune(2)
        assert pruned.get("A") == 5
        assert pruned.get("D") == 4
        assert pruned.get("B") == 0
        assert pruned.get("C") == 0

    def test_immutability(self):
        vc = VectorClock({"A": 1})
        vc.increment("A")
        assert vc.get("A") == 1  # Original unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
