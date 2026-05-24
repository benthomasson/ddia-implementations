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


# --- Spec example: full write/read/failure/handoff lifecycle ---

def test_spec_example_full_lifecycle():
    """Covers the exact example from the spec's Example Usage section."""
    store = make_store()

    # Normal write - goes to preferred replicas
    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is False
    assert len(result["hints_stored"]) == 0

    # Take down one preferred replica
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    # Write still succeeds with sloppy quorum - hint is stored
    result = store.put("user:1", "Alice Updated", current_time=2)
    assert result["success"] is True
    assert result["sloppy"] is True
    assert len(result["hints_stored"]) == 1
    assert result["hints_stored"][0]["target_node"] == preferred[0]

    # Hint is on a non-preferred node
    hint_node_id = result["hints_stored"][0]["hint_node"]
    assert hint_node_id not in preferred

    # Bring the node back and trigger handoff
    store.set_node_available(preferred[0], True)
    handoff = store.trigger_handoff(preferred[0], current_time=3)
    assert handoff["hints_delivered"] == 1
    assert "user:1" in handoff["keys_recovered"]

    # Hints cleaned up from hint node
    assert store.nodes[hint_node_id].hint_count() == 0

    # The recovered node now has the latest data
    read_result = store.get("user:1")
    assert read_result["value"] == "Alice Updated"
    assert read_result["version"] == 2


# --- Strict quorum rejects writes when preferred replicas insufficient ---

def test_strict_quorum_rejects_write():
    store = make_store(sloppy_quorum=False)
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)
    store.set_node_available(preferred[1], False)

    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is False
    assert result["sloppy"] is False
    assert len(result["hints_stored"]) == 0


# --- Sloppy quorum succeeds with non-preferred nodes ---

def test_sloppy_quorum_uses_substitutes():
    store = make_store()
    preferred = store.get_preferred_nodes("key:x")
    store.set_node_available(preferred[0], False)
    store.set_node_available(preferred[1], False)

    result = store.put("key:x", "val", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is True
    assert len(result["hints_stored"]) == 2
    targets = {h["target_node"] for h in result["hints_stored"]}
    assert targets == {preferred[0], preferred[1]}


# --- Hint TTL expiry ---

def test_hint_expiry():
    store = make_store(hint_ttl=10)
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)
    store.put("user:1", "Alice", current_time=1)

    # Not expired yet (t=5 < 1+10=11)
    assert store.expire_all_hints(current_time=5) == 0

    # Expired (t=11 >= 1+10=11)
    assert store.expire_all_hints(current_time=11) == 1

    # Handoff finds nothing since hint was expired
    store.set_node_available(preferred[0], True)
    handoff = store.trigger_handoff(preferred[0], current_time=12)
    assert handoff["hints_delivered"] == 0


# --- All preferred replicas down ---

def test_all_preferred_down():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    for nid in preferred:
        store.set_node_available(nid, False)

    result = store.put("user:1", "Alice", current_time=1)
    assert result["success"] is True
    assert result["sloppy"] is True
    assert len(result["hints_stored"]) == 3
    for h in result["hints_stored"]:
        assert h["hint_node"] not in preferred
        assert h["target_node"] in preferred


# --- Hint node crash before handoff (data loss) ---

def test_hint_node_crash_loses_data():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")
    store.set_node_available(preferred[0], False)

    result = store.put("user:1", "Alice", current_time=1)
    hint_node_id = result["hints_stored"][0]["hint_node"]

    # Simulate hint node crash - hints lost
    store.nodes[hint_node_id].hints.clear()
    store.set_node_available(preferred[0], True)

    handoff = store.trigger_handoff(preferred[0], current_time=2)
    assert handoff["hints_delivered"] == 0
    assert store.nodes[preferred[0]].get("user:1") is None


# --- Multiple writes, handoff delivers latest version ---

def test_handoff_delivers_latest_version():
    store = make_store()
    preferred = store.get_preferred_nodes("user:1")

    store.put("user:1", "v1", current_time=1)
    store.set_node_available(preferred[0], False)
    store.put("user:1", "v2", current_time=2)
    store.put("user:1", "v3", current_time=3)

    store.set_node_available(preferred[0], True)
    store.trigger_handoff(preferred[0], current_time=4)

    data = store.nodes[preferred[0]].get("user:1")
    assert data[0] == "v3"
    assert data[1] == 3

    read = store.get("user:1")
    assert read["value"] == "v3"
    assert read["version"] == 3


# --- Key mapping is deterministic ---

def test_key_mapping_deterministic():
    store = make_store()
    nodes_a = store.get_preferred_nodes("key:1")
    nodes_b = store.get_preferred_nodes("key:1")
    assert nodes_a == nodes_b
    assert len(nodes_a) == 3

    # Different keys should not all map to same node
    first_nodes = {store.get_preferred_nodes(f"k:{i}")[0] for i in range(50)}
    assert len(first_nodes) > 1
