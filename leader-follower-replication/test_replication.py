"""Tests for leader-follower replication."""

import pytest
from replication import LeaderNode, FollowerNode, ReadSession, ReplicationLogEntry


class TestBasicReplication:
    def test_writes_appear_on_sync_follower(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")

        lsn = leader.put("k1", "v1")
        assert lsn == 1
        assert f.get("k1") == "v1"
        assert f.current_lsn() == 1

    def test_writes_appear_on_async_follower(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="async")

        leader.put("k1", "v1")
        assert f.get("k1") == "v1"

    def test_multiple_writes(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")

        lsn1 = leader.put("a", "1")
        lsn2 = leader.put("b", "2")
        assert lsn1 == 1
        assert lsn2 == 2
        assert f.get("a") == "1"
        assert f.get("b") == "2"
        assert f.current_lsn() == 2

    def test_leader_read(self):
        leader = LeaderNode("leader")
        leader.put("x", "y")
        assert leader.get("x") == "y"
        assert leader.get("missing") is None


class TestDeleteReplication:
    def test_delete_on_leader(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")

        leader.put("k1", "v1")
        assert leader.get("k1") == "v1"
        assert f.get("k1") == "v1"

        lsn = leader.delete("k1")
        assert lsn == 2
        assert leader.get("k1") is None
        assert f.get("k1") is None

    def test_delete_nonexistent_key(self):
        leader = LeaderNode("leader")
        lsn = leader.delete("nope")
        assert lsn == 1
        assert leader.get("nope") is None


class TestFollowerCatchup:
    def test_offline_misses_writes(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="async")

        leader.put("k1", "v1")
        assert f.current_lsn() == 1

        f.go_offline()
        leader.put("k2", "v2")
        leader.put("k3", "v3")
        assert f.current_lsn() == 1
        assert f.get("k2") is None

    def test_catchup_on_reconnect(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="async")

        leader.put("k1", "v1")
        f.go_offline()
        leader.put("k2", "v2")
        leader.put("k3", "v3")

        f.come_online(leader)
        assert f.get("k2") == "v2"
        assert f.get("k3") == "v3"
        assert f.current_lsn() == 3

    def test_catchup_from_scratch(self):
        leader = LeaderNode("leader")
        leader.put("a", "1")
        leader.put("b", "2")

        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="async")
        # Follower missed all previous writes, catch up
        f.come_online(leader)
        assert f.get("a") == "1"
        assert f.get("b") == "2"
        assert f.current_lsn() == 2


class TestReadAtLSN:
    def test_read_at_lsn_success(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")

        lsn = leader.put("k1", "v1")
        result = f.read_at_lsn("k1", min_lsn=lsn)
        assert result == "v1"

    def test_read_at_lsn_timeout(self):
        f = FollowerNode("f1")
        with pytest.raises(TimeoutError):
            f.read_at_lsn("k1", min_lsn=100, timeout=0.05)

    def test_read_at_lsn_key_not_found(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")

        lsn = leader.put("k1", "v1")
        result = f.read_at_lsn("missing", min_lsn=lsn)
        assert result is None


class TestMonotonicReads:
    def test_session_reads_consistent(self):
        leader = LeaderNode("leader")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="sync")
        leader.add_follower(f2, sync_mode="sync")

        leader.put("k1", "v1")
        leader.put("k2", "v2")

        session = ReadSession([f1, f2])
        assert session.read("k1") == "v1"
        assert session.read("k2") == "v2"

    def test_session_skips_stale_follower(self):
        leader = LeaderNode("leader")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="sync")
        leader.add_follower(f2, sync_mode="async")

        leader.put("k1", "v1")

        # Both are at LSN 1
        session = ReadSession([f1, f2])
        session.read("k1")  # sets last_seen_lsn to 1

        # Now f2 goes offline, f1 gets more writes
        f2.go_offline()
        leader.put("k2", "v2")

        # Session should pick f1 (LSN 2) not f2 (LSN 1, but offline anyway)
        val = session.read("k2")
        assert val == "v2"


class TestReplicationLag:
    def test_lag_reporting(self):
        leader = LeaderNode("leader")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="sync")
        leader.add_follower(f2, sync_mode="async")

        leader.put("k1", "v1")
        leader.put("k2", "v2")

        status = leader.follower_status()
        assert status["f1"]["lag"] == 0
        assert status["f2"]["lag"] == 0

        f2.go_offline()
        leader.put("k3", "v3")

        status = leader.follower_status()
        assert status["f1"]["lag"] == 0
        assert status["f2"]["lag"] == 1

    def test_replication_lag_method(self):
        f = FollowerNode("f1")
        assert f.replication_lag(10) == 10


class TestFailover:
    def test_promote_follower_to_leader(self):
        leader = LeaderNode("leader")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="sync")
        leader.add_follower(f2, sync_mode="sync")

        leader.put("k1", "v1")
        leader.put("k2", "v2")

        # Promote f1
        new_leader = f1.promote_to_leader()
        assert new_leader.get("k1") == "v1"
        assert new_leader.get("k2") == "v2"
        assert new_leader.current_lsn() == 2

        # New leader accepts writes
        new_leader.add_follower(f2, sync_mode="async")
        lsn = new_leader.put("k3", "v3")
        assert lsn == 3
        assert new_leader.get("k3") == "v3"
        assert f2.get("k3") == "v3"

    def test_unreplicated_writes_lost(self):
        leader = LeaderNode("leader")
        f1 = FollowerNode("f1")
        leader.add_follower(f1, sync_mode="async")

        leader.put("k1", "v1")
        f1.go_offline()
        leader.put("k2", "secret")  # f1 never got this

        new_leader = f1.promote_to_leader()
        assert new_leader.get("k1") == "v1"
        assert new_leader.get("k2") is None  # lost write


class TestLogRetention:
    def test_log_trimmed(self):
        leader = LeaderNode("leader")
        leader._log_retention = 100

        for i in range(200):
            leader.put(f"k{i}", f"v{i}")

        assert len(leader._log) == 100
        assert leader._log[0].lsn == 101

    def test_catchup_within_retention(self):
        leader = LeaderNode("leader")
        leader._log_retention = 100
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="async")

        for i in range(50):
            leader.put(f"k{i}", f"v{i}")

        f.go_offline()
        for i in range(50, 80):
            leader.put(f"k{i}", f"v{i}")

        f.come_online(leader)
        assert f.current_lsn() == 80
        assert f.get("k79") == "v79"


class TestSemiSync:
    def test_semi_sync_promotion(self):
        leader = LeaderNode("leader")
        f1 = FollowerNode("f1")
        f2 = FollowerNode("f2")
        leader.add_follower(f1, sync_mode="semi_sync")
        leader.add_follower(f2, sync_mode="async")

        leader.put("k1", "v1")
        assert f1.get("k1") == "v1"

        # f1 goes offline, f2 should be promoted to semi_sync
        f1.go_offline()
        leader.put("k2", "v2")

        assert leader._followers["f2"]["mode"] == "semi_sync"
        assert f2.get("k2") == "v2"


class TestEdgeCases:
    def test_empty_database(self):
        leader = LeaderNode("leader")
        assert leader.get("anything") is None
        assert leader.current_lsn() == 0

    def test_write_then_immediate_read(self):
        leader = LeaderNode("leader")
        leader.put("k", "v")
        assert leader.get("k") == "v"

    def test_multiple_updates_same_key(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")

        leader.put("k", "v1")
        leader.put("k", "v2")
        leader.put("k", "v3")
        assert leader.get("k") == "v3"
        assert f.get("k") == "v3"

    def test_lsn_starts_at_1(self):
        leader = LeaderNode("leader")
        assert leader.put("k", "v") == 1

    def test_remove_follower(self):
        leader = LeaderNode("leader")
        f = FollowerNode("f1")
        leader.add_follower(f, sync_mode="sync")
        leader.remove_follower("f1")
        leader.put("k", "v")
        assert f.get("k") is None  # no longer receiving updates

    def test_log_entries_api(self):
        leader = LeaderNode("leader")
        leader.put("a", "1")
        leader.put("b", "2")
        leader.put("c", "3")

        entries = leader.get_log_entries(after_lsn=1)
        assert len(entries) == 2
        assert entries[0].lsn == 2
        assert entries[1].lsn == 3

        all_entries = leader.get_log_entries()
        assert len(all_entries) == 3
