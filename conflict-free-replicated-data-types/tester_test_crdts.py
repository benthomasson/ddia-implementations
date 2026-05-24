"""Tests for CRDT implementations - covers spec examples and key edge cases."""

import pytest
from copy import deepcopy
from crdts import GCounter, PNCounter, LWWRegister, ORSet, CRDTReplicaGroup


# --- G-Counter ---

class TestGCounter:
    def test_spec_example_sync_convergence(self):
        """Spec example: 3 replicas increment independently, sync to 15."""
        group = CRDTReplicaGroup(GCounter, ["A", "B", "C"])
        group.get_replica("A").increment(5)
        group.get_replica("B").increment(3)
        group.get_replica("C").increment(7)
        assert group.get_replica("A").value() == 5
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("A").value() == 15

    def test_merge_commutative_associative_idempotent(self):
        a, b, c = GCounter("A"), GCounter("B"), GCounter("C")
        a.increment(3); b.increment(5); c.increment(7)
        # Commutativity
        ab = deepcopy(a).merge(deepcopy(b))
        ba = deepcopy(b).merge(deepcopy(a))
        assert ab == ba
        # Associativity
        ab_c = deepcopy(ab).merge(deepcopy(c))
        a_bc = deepcopy(a).merge(deepcopy(b).merge(deepcopy(c)))
        assert ab_c == a_bc
        # Idempotency
        before = deepcopy(a)
        a.merge(deepcopy(a))
        assert a == before

    def test_negative_increment_rejected(self):
        g = GCounter("A")
        with pytest.raises(ValueError):
            g.increment(-1)


# --- PN-Counter ---

class TestPNCounter:
    def test_spec_example(self):
        """Spec example: X increments 10, Y decrements 3, merge to 7."""
        group = CRDTReplicaGroup(PNCounter, ["X", "Y"])
        group.get_replica("X").increment(10)
        group.get_replica("Y").decrement(3)
        group.sync_all()
        assert group.get_replica("X").value() == 7

    def test_negative_value(self):
        pn = PNCounter("A")
        pn.decrement(5)
        assert pn.value() == -5

    def test_mixed_operations_converge(self):
        group = CRDTReplicaGroup(PNCounter, ["A", "B", "C"])
        group.get_replica("A").increment(10)
        group.get_replica("A").decrement(2)
        group.get_replica("B").increment(5)
        group.get_replica("B").decrement(8)
        group.get_replica("C").decrement(1)
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("A").value() == 4  # 10-2+5-8-1


# --- LWW-Register ---

class TestLWWRegister:
    def test_spec_example_higher_timestamp_wins(self):
        """Spec example: R2's 'world' at ts=2.0 beats R1's 'hello' at ts=1.0."""
        group = CRDTReplicaGroup(LWWRegister, ["R1", "R2"])
        group.get_replica("R1").set("hello", timestamp=1.0)
        group.get_replica("R2").set("world", timestamp=2.0)
        group.sync_all()
        assert group.get_replica("R1").get() == "world"

    def test_deterministic_tiebreak(self):
        """Same timestamp: higher replica_id wins, both directions agree."""
        r1, r2 = LWWRegister("A"), LWWRegister("B")
        r1.set("from_A", timestamp=5.0)
        r2.set("from_B", timestamp=5.0)
        m1 = deepcopy(r1).merge(deepcopy(r2))
        m2 = deepcopy(r2).merge(deepcopy(r1))
        assert m1.get() == m2.get() == "from_B"  # "B" > "A"

    def test_sequential_auto_clock(self):
        r = LWWRegister("R1")
        r.set("first")
        r.set("second")
        r.set("third")
        assert r.get() == "third"


# --- OR-Set ---

class TestORSet:
    def test_spec_example_disjoint_merge(self):
        """Spec example: S1 adds apple, S2 adds banana, merge has both."""
        group = CRDTReplicaGroup(ORSet, ["S1", "S2"])
        group.get_replica("S1").add("apple")
        group.get_replica("S2").add("banana")
        group.sync_all()
        assert group.get_replica("S1").elements() == {"apple", "banana"}

    def test_spec_example_concurrent_add_wins(self):
        """Spec example: concurrent add wins over remove."""
        group = CRDTReplicaGroup(ORSet, ["S1", "S2"])
        group.get_replica("S1").add("apple")
        group.sync_all()
        # S1 removes, S2 concurrently re-adds
        group.get_replica("S1").remove("apple")
        group.get_replica("S2").add("apple")
        group.sync_all()
        assert group.get_replica("S1").contains("apple")
        assert group.get_replica("S2").contains("apple")

    def test_add_remove_add(self):
        s = ORSet("S1")
        s.add("item")
        s.remove("item")
        assert not s.contains("item")
        s.add("item")
        assert s.contains("item")

    def test_remove_only_known_tags(self):
        """Remove on one replica doesn't affect unknown tags from another."""
        s1, s2 = ORSet("S1"), ORSet("S2")
        s1.add("x")
        s2.add("x")
        s1.remove("x")
        s1.merge(deepcopy(s2))
        assert s1.contains("x")  # s2's tag survives

    def test_merge_commutative_associative_idempotent(self):
        a, b, c = ORSet("A"), ORSet("B"), ORSet("C")
        a.add("x"); a.add("y")
        b.add("y"); b.add("z")
        c.add("z"); c.add("w")
        # Commutativity
        ab = deepcopy(a).merge(deepcopy(b))
        ba = deepcopy(b).merge(deepcopy(a))
        assert ab == ba
        # Associativity
        ab_c = deepcopy(ab).merge(deepcopy(c))
        a_bc = deepcopy(a).merge(deepcopy(b).merge(deepcopy(c)))
        assert ab_c == a_bc
        # Idempotency
        before = deepcopy(a)
        a.merge(deepcopy(a))
        assert a == before


# --- ReplicaGroup ---

class TestReplicaGroup:
    def test_partial_sync_then_full(self):
        group = CRDTReplicaGroup(GCounter, ["A", "B", "C"])
        group.get_replica("A").increment(1)
        group.get_replica("B").increment(2)
        group.sync("A", "B")
        assert group.get_replica("B").value() == 3
        assert group.get_replica("C").value() == 0
        group.sync_all()
        assert group.all_converged()

    def test_many_replicas(self):
        ids = [f"R{i}" for i in range(15)]
        group = CRDTReplicaGroup(GCounter, ids)
        for i, rid in enumerate(ids):
            group.get_replica(rid).increment(i + 1)
        group.sync_all()
        assert group.all_converged()
        assert group.get_replica("R0").value() == 120  # sum(1..15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
