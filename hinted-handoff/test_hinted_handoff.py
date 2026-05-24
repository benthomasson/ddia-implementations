"""Tests for hinted handoff implementation."""
import pytest
from hinted_handoff import Hint, Node, HintedHandoffStore


def make_store(**kwargs):
    defaults = dict(
        node_ids=["A", "B", "C", "D", "E"],
        replication_factor=3,
        write_quorum=2,
        read_quorum=2,
        hint_ttl=50,
        sloppy_quorum=True,
    )
    defaults.update(kwargs)
    return HintedHandoffStore(**defaults)


# 1. Normal write/read with all nodes healthy
def test_normal_write_read():
    store = make_store()
    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is False
    assert len(result["hints_stored"]) == 0
    assert len(result["replicas_written"]) == 3

    read = store.get("user:1")
    assert read["value"] == "Alice"
    assert read["version"] == 1


# 2. Write to a down node creates a hint on a substitute node
def test_write_creates_hint_on_substitute():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is True
    assert len(result["hints_stored"]) == 1
    assert result["hints_stored"][0]["target_node"] == preferred[0]

    # Hint is on a non-preferred node
    hint_node_id = result["hints_stored"][0]["hint_node"]
    assert hint_node_id not in preferred
    assert store.nodes[hint_node_id].hint_count() == 1


# 3. Handoff delivers hints to recovered node
def test_handoff_delivers_hints():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    store.put("user:1", "Alice", current_time=1)
    store.set_node_available(preferred[0], True)

    handoff = store.trigger_handoff(preferred[0], current_time=2)
    assert handoff["hints_delivered"] == 1
    assert "user:1" in handoff["keys_recovered"]

    # Recovered node has the data
    result = store.nodes[preferred[0]].get("user:1")
    assert result is not None
    assert result[0] == "Alice"


# 4. Hints removed from hint node after successful handoff
def test_hints_removed_after_handoff():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    result = store.put("user:1", "Alice", current_time=1)
    hint_node_id = result["hints_stored"][0]["hint_node"]
    assert store.nodes[hint_node_id].hint_count() == 1

    store.set_node_available(preferred[0], True)
    store.trigger_handoff(preferred[0], current_time=2)

    assert store.nodes[hint_node_id].hint_count() == 0


# 5. Hint expiry removes old hints
def test_hint_expiry():
    store = make_store(hint_ttl=10)
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    store.put("user:1", "Alice", current_time=1)

    # Not expired yet
    expired = store.expire_all_hints(current_time=5)
    assert expired == 0

    # Now expired (created_at=1, ttl=10, so expires at t=11)
    expired = store.expire_all_hints(current_time=11)
    assert expired == 1


# 6. Sloppy quorum disabled: write fails if preferred replicas down
def test_strict_quorum_fails():
    store = make_store(sloppy_quorum=False)
    preferred = store.get_preferred_nodes("user:1")
    # Take down enough preferred replicas so W can't be met
    store.set_node_available(preferred[0], False)
    store.set_node_available(preferred[1], False)

    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is False
    assert result["sloppy"] is False


# 7. Sloppy quorum enabled: write succeeds using non-preferred nodes
def test_sloppy_quorum_succeeds():
    store = make_store(sloppy_quorum=True)
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)
    store.set_node_available(preferred[1], False)

    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is True
    assert len(result["hints_stored"]) == 2


# 8. Multiple concurrent failures with hints on multiple nodes
def test_multiple_failures():
    store = make_store()
    preferred = store.get_preferred_nodes("key:1")

    # Take down two preferred replicas
    store.set_node_available(preferred[0], False)
    store.set_node_available(preferred[1], False)

    result = store.put("key:1", "value1", current_time=1)
    assert result["success"] is True
    assert len(result["hints_stored"]) == 2

    targets = {h["target_node"] for h in result["hints_stored"]}
    assert targets == {preferred[0], preferred[1]}


# 9. Recovered node serves correct data after handoff
def test_recovered_node_serves_correct_data():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    store.put("user:1", "v1", current_time=1)
    store.put("user:1", "v2", current_time=2)

    store.set_node_available(preferred[0], True)
    store.trigger_handoff(preferred[0], current_time=3)

    data = store.nodes[preferred[0]].get("user:1")
    assert data[0] == "v2"
    assert data[1] == 2


# 10. Key-to-node mapping is deterministic and distributes evenly
def test_key_mapping_deterministic_and_distributed():
    store = make_store()

    # Deterministic
    for _ in range(10):
        assert store.get_preferred_nodes("key:1") == store.get_preferred_nodes("key:1")

    # Distribution: check that different keys map to different starting nodes
    node_counts = {}
    for i in range(100):
        preferred = store.get_preferred_nodes(f"key:{i}")
        first = preferred[0]
        node_counts[first] = node_counts.get(first, 0) + 1

    # Each node should get at least some keys (not all on one node)
    assert len(node_counts) > 1


# 11. All preferred replicas down, all hints go to other nodes
def test_all_preferred_down():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")

    for nid in preferred:
        store.set_node_available(nid, False)

    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is True

    # All hints should be on non-preferred nodes
    for h in result["hints_stored"]:
        assert h["hint_node"] not in preferred
        assert h["target_node"] in preferred


# 12. Hint node goes down before handoff (hints are lost)
def test_hint_node_down_before_handoff():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    result = store.put("user:1", "Alice", current_time=1)
    hint_node_id = result["hints_stored"][0]["hint_node"]

    # Simulate hint node crashing (hints are in memory, lost if node is down)
    # The hints still exist in memory but the node is unavailable
    store.set_node_available(hint_node_id, False)
    store.set_node_available(preferred[0], True)

    # Handoff still runs (coordinator iterates all nodes regardless),
    # but in a real system the hint node being down means no delivery.
    # Our in-memory model delivers anyway since hints are in memory.
    # The key insight: if hint node data is truly lost, recovered node misses data.
    # We simulate by clearing hints before handoff.
    store.nodes[hint_node_id].hints.clear()

    handoff = store.trigger_handoff(preferred[0], current_time=2)
    assert handoff["hints_delivered"] == 0

    # The recovered node doesn't have the data
    data = store.nodes[preferred[0]].get("user:1")
    assert data is None


# 13. Read after handoff returns the latest version
def test_read_after_handoff_returns_latest():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")

    # Write v1 normally
    store.put("user:1", "v1", current_time=1)

    # Take down a node, write v2
    store.set_node_available(preferred[0], False)
    store.put("user:1", "v2", current_time=2)

    # Recover and handoff
    store.set_node_available(preferred[0], True)
    store.trigger_handoff(preferred[0], current_time=3)

    # Read should return v2
    read = store.get("user:1")
    assert read["value"] == "v2"
    assert read["version"] == 2
