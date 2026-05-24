"""Tests for CRDT implementations covering all 20 required test scenarios."""

import pytest
from copy import deepcopy
from crdts import GCounter, PNCounter, LWWRegister, ORSet, CRDTReplicaGroup


# === 1. G-Counter: increment, value, merge, convergence ===
class TestGCounterBasic:
    def test_increment_and_value(self):
        g = GCounter("A")
        g.increment(5)
        assert g.value() == 5
        g.increment(3)
        assert g.value() == 8

    def test_merge_and_convergence(self):
        group = CRDTReplicaGroup(GCounter, ["A", "B", "C"])
        group.get_replica("A").increment(5)
        group.get_replica("B").increment(3)
        group.get_replica("C").increment(7)
        assert group.get_replica("A").value() == 5
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("A").value() == 15


# === 2. G-Counter: commutative, associative, idempotent ===
class TestGCounterProperties:
    def test_commutativity(self):
        a, b = GCounter("A"), GCounter("B")
        a.increment(3)
        b.increment(5)
        ab = deepcopy(a).merge(deepcopy(b))
        ba = deepcopy(b).merge(deepcopy(a))
        assert ab == ba

    def test_associativity(self):
        a, b, c = GCounter("A"), GCounter("B"), GCounter("C")
        a.increment(1); b.increment(2); c.increment(3)
        ab_c = deepcopy(deepcopy(a).merge(deepcopy(b))).merge(deepcopy(c))
        a_bc = deepcopy(a).merge(deepcopy(deepcopy(b).merge(deepcopy(c))))
        assert ab_c == a_bc

    def test_idempotency(self):
        a = GCounter("A")
        a.increment(5)
        before = deepcopy(a)
        a.merge(deepcopy(a))
        assert a == before


# === 3. G-Counter: concurrent increments converge ===
class TestGCounterConcurrent:
    def test_concurrent_increments(self):
        group = CRDTReplicaGroup(GCounter, ["A", "B", "C"])
        for _ in range(10):
            group.get_replica("A").increment()
        for _ in range(20):
            group.get_replica("B").increment()
        for _ in range(30):
            group.get_replica("C").increment()
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("A").value() == 60


# === 4. PN-Counter: increment, decrement, negative values ===
class TestPNCounterBasic:
    def test_increment_decrement(self):
        pn = PNCounter("A")
        pn.increment(10)
        assert pn.value() == 10
        pn.decrement(3)
        assert pn.value() == 7

    def test_negative_value(self):
        pn = PNCounter("A")
        pn.decrement(5)
        assert pn.value() == -5


# === 5. PN-Counter: merge concurrent increments and decrements ===
class TestPNCounterMerge:
    def test_merge_concurrent(self):
        group = CRDTReplicaGroup(PNCounter, ["X", "Y"])
        group.get_replica("X").increment(10)
        group.get_replica("Y").decrement(3)
        group.sync_all()
        assert group.get_replica("X").value() == 7
        assert group.get_replica("Y").value() == 7


# === 6. PN-Counter: convergence with mixed operations ===
class TestPNCounterConvergence:
    def test_mixed_operations_converge(self):
        group = CRDTReplicaGroup(PNCounter, ["A", "B", "C"])
        group.get_replica("A").increment(10)
        group.get_replica("A").decrement(2)
        group.get_replica("B").increment(5)
        group.get_replica("B").decrement(8)
        group.get_replica("C").decrement(1)
        group.sync_all()
        assert group.all_converged()
        # 10 - 2 + 5 - 8 - 1 = 4
        assert group.get_replica("A").value() == 4


# === 7. LWW-Register: higher timestamp wins ===
class TestLWWRegisterTimestamp:
    def test_higher_timestamp_wins(self):
        group = CRDTReplicaGroup(LWWRegister, ["R1", "R2"])
        group.get_replica("R1").set("hello", timestamp=1.0)
        group.get_replica("R2").set("world", timestamp=2.0)
        group.sync_all()
        assert group.get_replica("R1").get() == "world"
        assert group.get_replica("R2").get() == "world"


# === 8. LWW-Register: deterministic tiebreaking ===
class TestLWWRegisterTiebreak:
    def test_tiebreak_by_replica_id(self):
        r1, r2 = LWWRegister("A"), LWWRegister("B")
        r1.set("from_A", timestamp=5.0)
        r2.set("from_B", timestamp=5.0)
        m1 = deepcopy(r1).merge(deepcopy(r2))
        m2 = deepcopy(r2).merge(deepcopy(r1))
        # Both should agree (higher replica_id wins)
        assert m1.get() == m2.get()
        assert m1.get() == "from_B"  # "B" > "A"


# === 9. LWW-Register: sequential updates on same replica ===
class TestLWWRegisterSequential:
    def test_sequential_updates(self):
        r = LWWRegister("R1")
        r.set("first")
        r.set("second")
        r.set("third")
        assert r.get() == "third"
        assert r.get_timestamp() == 3.0


# === 10. LWW-Register: concurrent updates on different replicas ===
class TestLWWRegisterConcurrent:
    def test_concurrent_updates(self):
        group = CRDTReplicaGroup(LWWRegister, ["R1", "R2", "R3"])
        group.get_replica("R1").set("val1", timestamp=1.0)
        group.get_replica("R2").set("val2", timestamp=3.0)
        group.get_replica("R3").set("val3", timestamp=2.0)
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("R1").get() == "val2"


# === 11. OR-Set: add, remove, contains, elements ===
class TestORSetBasic:
    def test_add_remove_contains(self):
        s = ORSet("S1")
        s.add("apple")
        s.add("banana")
        assert s.contains("apple")
        assert s.elements() == {"apple", "banana"}
        assert s.remove("apple") is True
        assert not s.contains("apple")
        assert s.elements() == {"banana"}
        assert s.remove("cherry") is False


# === 12. OR-Set: concurrent add wins over concurrent remove ===
class TestORSetAddWins:
    def test_concurrent_add_wins(self):
        group = CRDTReplicaGroup(ORSet, ["S1", "S2"])
        group.get_replica("S1").add("apple")
        group.sync_all()
        # S1 removes, S2 concurrently re-adds
        group.get_replica("S1").remove("apple")
        group.get_replica("S2").add("apple")
        group.sync_all()
        assert group.get_replica("S1").contains("apple")
        assert group.get_replica("S2").contains("apple")


# === 13. OR-Set: remove only removes known tags ===
class TestORSetRemoveKnownTags:
    def test_remove_only_known(self):
        s1, s2 = ORSet("S1"), ORSet("S2")
        s1.add("x")
        s2.add("x")
        # s1 removes only its own tag for "x"
        s1.remove("x")
        s1.merge(deepcopy(s2))
        # s2's tag for "x" survives
        assert s1.contains("x")


# === 14. OR-Set: add-remove-add results in element present ===
class TestORSetAddRemoveAdd:
    def test_add_remove_add(self):
        s = ORSet("S1")
        s.add("item")
        s.remove("item")
        assert not s.contains("item")
        s.add("item")
        assert s.contains("item")


# === 15. OR-Set: merge of disjoint sets ===
class TestORSetDisjoint:
    def test_disjoint_merge(self):
        group = CRDTReplicaGroup(ORSet, ["S1", "S2"])
        group.get_replica("S1").add("apple")
        group.get_replica("S2").add("banana")
        group.sync_all()
        assert group.get_replica("S1").elements() == {"apple", "banana"}
        assert group.get_replica("S2").elements() == {"apple", "banana"}


# === 16. OR-Set: commutative, associative, idempotent ===
class TestORSetProperties:
    def test_commutativity(self):
        a, b = ORSet("A"), ORSet("B")
        a.add("x"); a.add("y")
        b.add("y"); b.add("z")
        ab = deepcopy(a).merge(deepcopy(b))
        ba = deepcopy(b).merge(deepcopy(a))
        assert ab.elements() == ba.elements()
        assert ab == ba

    def test_associativity(self):
        a, b, c = ORSet("A"), ORSet("B"), ORSet("C")
        a.add("x"); b.add("y"); c.add("z")
        ab_c = deepcopy(deepcopy(a).merge(deepcopy(b))).merge(deepcopy(c))
        a_bc = deepcopy(a).merge(deepcopy(deepcopy(b).merge(deepcopy(c))))
        assert ab_c.elements() == a_bc.elements()
        assert ab_c == a_bc

    def test_idempotency(self):
        a = ORSet("A")
        a.add("x"); a.add("y")
        before = deepcopy(a)
        a.merge(deepcopy(a))
        assert a == before


# === 17. ReplicaGroup: sync_all convergence ===
class TestReplicaGroupConvergence:
    def test_sync_all_converges(self):
        for CRDTClass in [GCounter, PNCounter]:
            group = CRDTReplicaGroup(CRDTClass, ["A", "B", "C", "D"])
            group.get_replica("A").increment(1)
            group.get_replica("B").increment(2)
            group.get_replica("C").increment(3)
            group.get_replica("D").increment(4)
            group.sync_all()
            assert group.all_converged()


# === 18. ReplicaGroup: partial sync and eventual convergence ===
class TestReplicaGroupPartialSync:
    def test_partial_then_full(self):
        group = CRDTReplicaGroup(GCounter, ["A", "B", "C"])
        group.get_replica("A").increment(1)
        group.get_replica("B").increment(2)
        # Only sync A -> B
        group.sync("A", "B")
        assert group.get_replica("B").value() == 3
        assert group.get_replica("C").value() == 0
        # Now sync all
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("C").value() == 3


# === 19. ReplicaGroup: 10+ replicas ===
class TestReplicaGroupManyReplicas:
    def test_many_replicas(self):
        ids = [f"R{i}" for i in range(15)]
        group = CRDTReplicaGroup(GCounter, ids)
        for i, rid in enumerate(ids):
            group.get_replica(rid).increment(i + 1)
        group.sync_all()
        assert group.all_converged()
        # Sum of 1..15 = 120
        assert group.get_replica("R0").value() == 120


# === 20. All CRDTs: serialization round-trip via state() ===
class TestSerialization:
    def test_gcounter_state(self):
        g = GCounter("A")
        g.increment(5)
        s = g.state()
        assert s == {"counts": {"A": 5}}

    def test_pncounter_state(self):
        pn = PNCounter("A")
        pn.increment(10)
        pn.decrement(3)
        s = pn.state()
        assert s["p"]["counts"]["A"] == 10
        assert s["n"]["counts"]["A"] == 3

    def test_lww_state(self):
        r = LWWRegister("R1")
        r.set("hello", timestamp=1.0)
        s = r.state()
        assert s["value"] == "hello"
        assert s["timestamp"] == 1.0

    def test_orset_state(self):
        s = ORSet("S1")
        s.add("x")
        state = s.state()
        assert "x" in state["elements"]
        assert len(state["elements"]["x"]) == 1

    def test_lww_json_values(self):
        """LWW-Register supports any JSON-serializable value."""
        r = LWWRegister("R1")
        for val in [42, "string", [1, 2, 3], {"key": "value"}, None, True]:
            r.set(val)
            assert r.get() == val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
