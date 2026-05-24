"""Tests for leader-follower replication."""

import pytest
from replication import LeaderNode, FollowerNode, ReadSession, ReplicationLogEntry


class TestSpecExample:
    """Tests based on the example usage from the spec."""

    def test_full_example_flow(self):
        """End-to-end test matching the spec's example usage."""
        leader = LeaderNode("leader-1")
        follower1 = FollowerNode("follower-1")
        follower2 = FollowerNode("follower-2")

        leader.add_follower(follower1, sync_mode="sync")
        leader.add_follower(follower2, sync_mode="async")

        # Writes
        lsn1 = leader.put("user:1", "alice")
        lsn2 = leader.put("user:2", "bob")
        assert lsn1 == 1
        assert lsn2 == 2

        # Sync follower is up-to-date
        assert follower1.get("user:1") == "alice"
        assert follower1.current_lsn() == 2

        # Async follower also up-to-date in single-threaded sim
        assert follower2.get("user:1") == "alice"

        # Delete
        lsn3 = leader.delete("user:1")
        assert leader.get("user:1") is None
        assert follower1.get("user:1") is None

        # Follower failure and catchup
        follower2.go_offline()
        leader.put("user:3", "charlie")
        leader.put("user:4", "diana")
        assert follower2.current_lsn() == 3  # last applied before offline

        follower2.come_online(leader)
        assert follower2.get("user:3") == "charlie"
        assert follower2.get("user:4") == "diana"
        assert follower2.current_lsn() == 5

        # Replication lag
        status = leader.follower_status()
        assert status["follower-1"]["lag"] == 0
        assert status["follower-2"]["lag"] == 0

        # Read-your-writes
        lsn = leader.put("user:5", "eve")
        result = follower1.read_at_lsn("user:5", min_lsn=lsn)
        assert result == "eve"

        # Monotonic reads
        session = ReadSession([follower1, follower2])
        val = session.read("user:2")
        assert val == "bob"

        # Failover
        new_leader = follower1.promote_to_leader()
        new_leader.add_follower(follower2, sync_mode="async")
        new_leader.put("user:6", "frank")
        assert new_leader.get("user:6") == "frank"


class TestReplicationBasics:
    def test_sync_follower_always_current(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="sync")

        for i in range(10):
            lsn = leader.put(f"k{i}", f"v{i}")
            assert f.current_lsn() == lsn

    def test_delete_propagates_to_follower(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="sync")

        leader.put("x", "1")
        leader.delete("x")
        assert f.get("x") is None
        assert f.current_lsn() == 2


class TestFollowerCatchup:
    def test_catchup_from_scratch(self):
        """Follower added after writes catches up via come_online."""
        leader = LeaderNode("l")
        leader.put("a", "1")
        leader.put("b", "2")
        leader.put("c", "3")

        f = FollowerNode("f")
        leader.add_follower(f)
        f.come_online(leader)
        assert f.get("a") == "1"
        assert f.get("c") == "3"
        assert f.current_lsn() == 3

    def test_offline_follower_misses_then_catches_up(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="async")

        leader.put("k1", "v1")
        f.go_offline()
        leader.put("k2", "v2")
        leader.put("k3", "v3")

        assert f.current_lsn() == 1
        assert f.get("k2") is None

        f.come_online(leader)
        assert f.current_lsn() == 3
        assert f.get("k2") == "v2"


class TestReadAtLSN:
    def test_read_at_lsn_succeeds_when_caught_up(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="sync")

        lsn = leader.put("k", "v")
        assert f.read_at_lsn("k", min_lsn=lsn) == "v"

    def test_read_at_lsn_times_out_when_behind(self):
        f = FollowerNode("f")
        with pytest.raises(TimeoutError):
            f.read_at_lsn("k", min_lsn=5, timeout=0.05)


class TestMonotonicReads:
    def test_session_never_reads_stale(self):
        leader = LeaderNode("l")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="sync")
        leader.add_follower(f2, sync_mode="sync")

        leader.put("k1", "v1")
        leader.put("k2", "v2")

        session = ReadSession([f1, f2])
        session.read("k1")  # advances last_seen_lsn to 2

        # Make f2 lag by taking it offline
        f2.go_offline()
        leader.put("k3", "v3")

        # Session should pick f1 (LSN 3) not f2 (LSN 2, offline)
        assert session.read("k3") == "v3"


class TestFailover:
    def test_promoted_follower_accepts_writes(self):
        leader = LeaderNode("l")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="sync")
        leader.add_follower(f2, sync_mode="sync")

        leader.put("k1", "v1")
        leader.put("k2", "v2")

        new_leader = f1.promote_to_leader()
        assert new_leader.get("k1") == "v1"
        assert new_leader.current_lsn() == 2

        new_leader.add_follower(f2, sync_mode="async")
        lsn = new_leader.put("k3", "v3")
        assert lsn == 3
        assert f2.get("k3") == "v3"

    def test_unreplicated_writes_lost_on_failover(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="async")

        leader.put("k1", "v1")
        f.go_offline()
        leader.put("k2", "lost")

        new_leader = f.promote_to_leader()
        assert new_leader.get("k1") == "v1"
        assert new_leader.get("k2") is None


class TestSemiSync:
    def test_semi_sync_promotes_async_on_failure(self):
        leader = LeaderNode("l")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="semi_sync")
        leader.add_follower(f2, sync_mode="async")

        leader.put("k1", "v1")
        f1.go_offline()
        leader.put("k2", "v2")

        assert leader._followers["f2"]["mode"] == "semi_sync"
        assert f2.get("k2") == "v2"


class TestEdgeCases:
    def test_empty_database(self):
        leader = LeaderNode("l")
        assert leader.get("x") is None
        assert leader.current_lsn() == 0

    def test_overwrite_same_key(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="sync")

        leader.put("k", "v1")
        leader.put("k", "v2")
        leader.put("k", "v3")
        assert leader.get("k") == "v3"
        assert f.get("k") == "v3"
        assert f.current_lsn() == 3

    def test_log_retention_trims(self):
        leader = LeaderNode("l")
        leader._log_retention = 50
        for i in range(100):
            leader.put(f"k{i}", f"v{i}")
        assert len(leader._log) == 50
        assert leader._log[0].lsn == 51

    def test_replication_lag_reporting(self):
        leader = LeaderNode("l")
        f = FollowerNode("f")
        leader.add_follower(f, sync_mode="async")

        leader.put("k1", "v1")
        f.go_offline()
        leader.put("k2", "v2")
        leader.put("k3", "v3")

        assert f.replication_lag(leader.current_lsn()) == 2
        status = leader.follower_status()
        assert status["f"]["lag"] == 2
