"""Tests for fencing tokens implementation."""
import pytest
from fencing_tokens import (
    FencingToken, LockService, FencedResourceServer,
    UnfencedResourceServer, Client,
)


class TestLockBasics:
    def test_acquire_returns_valid_token(self):
        ls = LockService()
        token = ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        assert token is not None
        assert token.token == 1
        assert token.lock_name == "lock-1"
        assert token.client_id == "client-A"

    def test_tokens_globally_monotonic(self):
        ls = LockService()
        t1 = ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        t2 = ls.acquire("lock-2", "client-B", current_time=0, ttl=10)
        t3 = ls.acquire("lock-1", "client-C", current_time=11, ttl=10)
        assert t1.token < t2.token < t3.token

    def test_mutual_exclusion(self):
        ls = LockService()
        ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        assert ls.acquire("lock-1", "client-B", current_time=1, ttl=10) is None

    def test_lock_expiry_allows_reacquire(self):
        ls = LockService()
        ls.acquire("lock-1", "client-A", current_time=0, ttl=5)
        token = ls.acquire("lock-1", "client-B", current_time=5, ttl=5)
        assert token is not None
        assert token.client_id == "client-B"

    def test_renewal_extends_ttl_same_token(self):
        ls = LockService()
        token = ls.acquire("lock-1", "client-A", current_time=0, ttl=5)
        original_value = token.token
        assert ls.renew("lock-1", "client-A", current_time=3, ttl=10)
        # Token value unchanged
        assert token.token == original_value
        # Lock should now expire at time 13, not 5
        assert not token.is_expired(12)
        assert token.is_expired(13)


class TestFencingScenarios:
    def test_unsafe_scenario_stale_write_corrupts(self):
        """Without fencing, a stale client overwrites newer data."""
        ls = LockService()
        unfenced = UnfencedResourceServer()
        c_a = Client("client-A", ls)
        c_b = Client("client-B", ls)

        c_a.acquire_lock("res-lock", current_time=0, ttl=5)
        unfenced.write("shared", "val", "written-by-A")

        # A pauses, lock expires, B acquires and writes
        c_b.acquire_lock("res-lock", current_time=6, ttl=5)
        unfenced.write("shared", "val", "written-by-B")

        # A wakes up, stale write succeeds -- corruption
        unfenced.write("shared", "val", "stale-write-by-A")
        assert unfenced.read("shared", "val") == "stale-write-by-A"

    def test_safe_scenario_stale_write_rejected(self):
        """With fencing, a stale client's write is rejected."""
        ls = LockService()
        fenced = FencedResourceServer()
        c_a = Client("client-A", ls)
        c_b = Client("client-B", ls)

        t_a = c_a.acquire_lock("res-lock", current_time=0, ttl=5)
        result = c_a.write_to_resource(fenced, "shared", "val", "written-by-A", "res-lock")
        assert result["success"] is True

        # A pauses, lock expires, B acquires (higher token) and writes
        t_b = c_b.acquire_lock("res-lock", current_time=6, ttl=5)
        assert t_b.token > t_a.token
        result = c_b.write_to_resource(fenced, "shared", "val", "written-by-B", "res-lock")
        assert result["success"] is True

        # A tries stale write -- rejected
        result = c_a.write_to_resource(fenced, "shared", "val", "stale-write-by-A", "res-lock")
        assert result["success"] is False
        assert fenced.read("shared", "val") == "written-by-B"

    def test_fenced_server_rejects_lower_token(self):
        fenced = FencedResourceServer()
        fenced.write("r1", "k", "v1", fencing_token=5)
        result = fenced.write("r1", "k", "v2", fencing_token=3)
        assert result["success"] is False
        assert fenced.read("r1", "k") == "v1"

    def test_independent_resource_token_tracking(self):
        fenced = FencedResourceServer()
        fenced.write("r1", "k", "v1", fencing_token=10)
        # Different resource — token 2 should be accepted
        result = fenced.write("r2", "k", "v2", fencing_token=2)
        assert result["success"] is True

    def test_client_write_without_lock_fails(self):
        ls = LockService()
        fenced = FencedResourceServer()
        client = Client("c1", ls)
        result = client.write_to_resource(fenced, "r1", "k", "v", "no-lock")
        assert result["success"] is False
        assert "does not hold" in result["error"]

    def test_released_lock_token_still_valid_at_server(self):
        """Fencing tokens don't expire at the server — only locks expire."""
        ls = LockService()
        fenced = FencedResourceServer()
        client = Client("c1", ls)
        client.acquire_lock("lk", current_time=0, ttl=5)
        # Write with token 1
        result = client.write_to_resource(fenced, "r1", "k", "v1", "lk")
        assert result["success"] is True
        # Release the lock
        client.release_lock("lk")
        # The token 1 is still the highest seen — a direct write with token 1 still works
        result = fenced.write("r1", "k", "v2", fencing_token=1)
        assert result["success"] is True
