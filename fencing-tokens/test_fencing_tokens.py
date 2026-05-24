"""Tests for fencing tokens implementation."""
import pytest
from fencing_tokens import FencingToken, LockService, FencedResourceServer, UnfencedResourceServer, Client


class TestLockAcquisition:
    """Tests 1, 2, 3, 4, 5, 15."""

    def test_acquire_returns_valid_token(self):
        """Test 1: lock acquisition returns a valid fencing token."""
        ls = LockService()
        token = ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        assert token is not None
        assert token.token == 1
        assert token.lock_name == "lock-1"
        assert token.client_id == "client-A"

    def test_tokens_globally_monotonically_increasing(self):
        """Test 2: tokens are globally monotonically increasing."""
        ls = LockService()
        t1 = ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        t2 = ls.acquire("lock-2", "client-B", current_time=0, ttl=10)
        # Expire lock-1 so client-C can acquire it
        t3 = ls.acquire("lock-1", "client-C", current_time=11, ttl=10)
        assert t1.token < t2.token < t3.token

    def test_mutual_exclusion(self):
        """Test 3: second client cannot acquire a held lock."""
        ls = LockService()
        ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        result = ls.acquire("lock-1", "client-B", current_time=1, ttl=10)
        assert result is None

    def test_lock_expiry(self):
        """Test 4: after TTL, another client can acquire the lock."""
        ls = LockService()
        ls.acquire("lock-1", "client-A", current_time=0, ttl=5)
        token = ls.acquire("lock-1", "client-B", current_time=5, ttl=5)
        assert token is not None
        assert token.client_id == "client-B"

    def test_release_and_reacquire(self):
        """Test 5: lock release and re-acquisition."""
        ls = LockService()
        ls.acquire("lock-1", "client-A", current_time=0, ttl=10)
        assert ls.release("lock-1", "client-A") is True
        token = ls.acquire("lock-1", "client-B", current_time=1, ttl=10)
        assert token is not None
        assert token.client_id == "client-B"

    def test_multiple_acquisitions_different_tokens(self):
        """Test 15: multiple lock acquisitions generate different tokens."""
        ls = LockService()
        t1 = ls.acquire("lock-1", "client-A", current_time=0, ttl=5)
        ls.release("lock-1", "client-A")
        t2 = ls.acquire("lock-1", "client-A", current_time=1, ttl=5)
        ls.release("lock-1", "client-A")
        t3 = ls.acquire("lock-1", "client-B", current_time=2, ttl=5)
        assert len({t1.token, t2.token, t3.token}) == 3


class TestLockRenewal:
    """Test 6."""

    def test_renewal_extends_ttl_without_changing_token(self):
        """Test 6: renewal extends TTL without changing the token."""
        ls = LockService()
        token = ls.acquire("lock-1", "client-A", current_time=0, ttl=5)
        original_token_value = token.token
        assert ls.renew("lock-1", "client-A", current_time=3, ttl=10) is True
        # Token number unchanged
        assert token.token == original_token_value
        # Lock is still held after original TTL would have expired
        info = ls.is_held("lock-1", current_time=8)
        assert info is not None
        assert info['client_id'] == "client-A"
        # Counter not incremented
        assert ls.get_token_counter() == 2  # only 1 acquire happened


class TestFencedServer:
    """Tests 7, 8, 11."""

    def test_accepts_valid_token(self):
        """Test 7: fenced server accepts valid token."""
        server = FencedResourceServer()
        result = server.write("res", "key", "value", fencing_token=1)
        assert result['success'] is True

    def test_rejects_stale_token(self):
        """Test 8: fenced server rejects stale (lower) token."""
        server = FencedResourceServer()
        server.write("res", "key", "value1", fencing_token=5)
        result = server.write("res", "key", "value2", fencing_token=3)
        assert result['success'] is False
        assert server.read("res", "key") == "value1"

    def test_independent_resource_token_tracking(self):
        """Test 11: multiple resources with independent token tracking."""
        server = FencedResourceServer()
        server.write("res-A", "key", "v1", fencing_token=5)
        # Different resource can use a lower token
        result = server.write("res-B", "key", "v2", fencing_token=2)
        assert result['success'] is True


class TestScenarios:
    """Tests 9, 10."""

    def test_unsafe_scenario_stale_write_corrupts(self):
        """Test 9: stale write corrupts unfenced resource."""
        ls = LockService()
        unfenced = UnfencedResourceServer()
        c_a = Client("client-A", ls)
        c_b = Client("client-B", ls)

        c_a.acquire_lock("lock", current_time=0, ttl=5)
        unfenced.write("shared", "value", "written-by-A")

        # Lock expires, B acquires
        c_b.acquire_lock("lock", current_time=6, ttl=5)
        unfenced.write("shared", "value", "written-by-B")

        # A wakes up, stale write succeeds -- corruption!
        unfenced.write("shared", "value", "stale-write-by-A")
        assert unfenced.read("shared", "value") == "stale-write-by-A"

    def test_safe_scenario_stale_write_rejected(self):
        """Test 10: stale write is rejected by fenced resource."""
        ls = LockService()
        fenced = FencedResourceServer()
        c_a = Client("client-A", ls)
        c_b = Client("client-B", ls)

        c_a.acquire_lock("lock", current_time=0, ttl=5)
        result = c_a.write_to_resource(fenced, "shared", "value", "written-by-A", "lock")
        assert result['success'] is True

        # Lock expires, B acquires (higher token)
        c_b.acquire_lock("lock", current_time=6, ttl=5)
        result = c_b.write_to_resource(fenced, "shared", "value", "written-by-B", "lock")
        assert result['success'] is True

        # A tries to write with stale token -- rejected!
        result = c_a.write_to_resource(fenced, "shared", "value", "stale-write-by-A", "lock")
        assert result['success'] is False
        assert fenced.read("shared", "value") == "written-by-B"


class TestClientEdgeCases:
    """Tests 12, 13, 14."""

    def test_write_without_holding_lock(self):
        """Test 12: client write without holding a lock fails gracefully."""
        ls = LockService()
        fenced = FencedResourceServer()
        client = Client("client-A", ls)
        result = client.write_to_resource(fenced, "res", "key", "val", "no-such-lock")
        assert result['success'] is False
        assert 'does not hold' in result['error']

    def test_concurrent_lock_attempts(self):
        """Test 13: concurrent lock attempts by multiple clients."""
        ls = LockService()
        c_a = Client("client-A", ls)
        c_b = Client("client-B", ls)
        c_c = Client("client-C", ls)

        t_a = c_a.acquire_lock("lock", current_time=0, ttl=10)
        t_b = c_b.acquire_lock("lock", current_time=0, ttl=10)
        t_c = c_c.acquire_lock("lock", current_time=0, ttl=10)

        assert t_a is not None
        assert t_b is None
        assert t_c is None

    def test_released_lock_token_still_valid_at_server(self):
        """Test 14: released lock's token is still valid at resource server."""
        ls = LockService()
        fenced = FencedResourceServer()
        client = Client("client-A", ls)

        token = client.acquire_lock("lock", current_time=0, ttl=10)
        result = client.write_to_resource(fenced, "res", "key", "v1", "lock")
        assert result['success'] is True

        client.release_lock("lock")

        # Token was 1; writing directly with that token should still work
        # (fencing tokens don't expire at the server)
        result = fenced.write("res", "key", "v2", fencing_token=token.token)
        assert result['success'] is True
