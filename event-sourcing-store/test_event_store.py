"""Tests for Event Sourcing Store."""
import os
import tempfile
import pytest

from event_store import EventStore, Projection, LiveProjection, ConcurrencyConflict, reconstruct_state


# Shared handlers
def on_opened(state, event):
    state[event.stream_id] = event.data.get("initial_balance", 0)

def on_deposited(state, event):
    state[event.stream_id] = state.get(event.stream_id, 0) + event.data["amount"]

def on_withdrawn(state, event):
    state[event.stream_id] = state.get(event.stream_id, 0) - event.data["amount"]

HANDLERS = {
    "AccountOpened": on_opened,
    "MoneyDeposited": on_deposited,
    "MoneyWithdrawn": on_withdrawn,
}

def make_projection(name, store):
    p = Projection(name, store)
    p.when("AccountOpened", on_opened)
    p.when("MoneyDeposited", on_deposited)
    p.when("MoneyWithdrawn", on_withdrawn)
    return p


def _seed_account(store):
    """Append the standard 4 events from the spec example."""
    store.append("account:1", "AccountOpened", {"owner": "Alice", "initial_balance": 0})
    store.append("account:1", "MoneyDeposited", {"amount": 100})
    store.append("account:1", "MoneyWithdrawn", {"amount": 30})
    store.append("account:1", "MoneyDeposited", {"amount": 50})


# --- 1. Basic append and read ---
def test_append_and_read():
    store = EventStore()
    _seed_account(store)
    events = store.read_stream("account:1")
    assert len(events) == 4
    assert events[0].event_type == "AccountOpened"
    assert events[0].event_id == 1
    assert events[-1].event_id == 4
    assert store.stream_version("account:1") == 4
    assert store.global_position == 4


# --- 2. Stream isolation ---
def test_stream_isolation():
    store = EventStore()
    store.append("a", "X", {"v": 1})
    store.append("b", "Y", {"v": 2})
    store.append("a", "X", {"v": 3})
    assert len(store.read_stream("a")) == 2
    assert len(store.read_stream("b")) == 1
    assert set(store.all_stream_ids()) == {"a", "b"}


# --- 3. Optimistic concurrency ---
def test_optimistic_concurrency():
    store = EventStore()
    _seed_account(store)
    # Correct version succeeds
    store.append("account:1", "MoneyDeposited", {"amount": 10}, expected_version=4)
    assert store.stream_version("account:1") == 5
    # Wrong version fails
    with pytest.raises(ConcurrencyConflict):
        store.append("account:1", "MoneyDeposited", {"amount": 5}, expected_version=3)


# --- 4. Batch append ---
def test_batch_append():
    store = EventStore()
    result = store.append_batch("acc:1", [
        ("Opened", {"x": 1}),
        ("Deposited", {"amount": 50}),
        ("Deposited", {"amount": 25}),
    ])
    assert len(result) == 3
    assert store.stream_version("acc:1") == 3
    assert [e.event_id for e in result] == [1, 2, 3]


# --- 5 & 6. Projection catch_up and incremental update ---
def test_projection_catch_up_and_incremental():
    store = EventStore()
    _seed_account(store)
    proj = make_projection("bal", store)
    processed = proj.catch_up()
    assert processed == 4
    assert proj.state["account:1"] == 120
    assert proj.position == 4

    # Add more, catch up again — only new events processed
    store.append("account:1", "MoneyDeposited", {"amount": 10})
    processed2 = proj.catch_up()
    assert processed2 == 1
    assert proj.state["account:1"] == 130


# --- 7. Temporal query ---
def test_temporal_query():
    store = EventStore()
    _seed_account(store)
    # After event 2: AccountOpened(0) + MoneyDeposited(100) = 100
    past = reconstruct_state(store, "account:1", HANDLERS, up_to=2)
    assert past["account:1"] == 100
    # After event 3: 100 - 30 = 70
    past3 = reconstruct_state(store, "account:1", HANDLERS, up_to=3)
    assert past3["account:1"] == 70


# --- 8 & 9. Snapshot save/load and subsequent events ---
def test_snapshot_save_load_and_resume():
    store = EventStore()
    _seed_account(store)
    store.append("account:1", "MoneyDeposited", {"amount": 10})  # event 5, balance=130

    proj = make_projection("bal", store)
    proj.catch_up()
    assert proj.state["account:1"] == 130
    proj.save_snapshot()

    # New projection loads snapshot
    p2 = make_projection("bal", store)
    assert p2.load_snapshot() is True
    assert p2.state["account:1"] == 130
    assert p2.position == 5

    # More events after snapshot
    store.append("account:1", "MoneyWithdrawn", {"amount": 20})
    p2.catch_up()
    assert p2.state["account:1"] == 110


# --- 10. Live projection ---
def test_live_projection():
    store = EventStore()
    _seed_account(store)
    live = LiveProjection("live_bal", store)
    live.when("AccountOpened", on_opened)
    live.when("MoneyDeposited", on_deposited)
    live.when("MoneyWithdrawn", on_withdrawn)
    live.catch_up()
    assert live.state["account:1"] == 120

    store.append("account:1", "MoneyDeposited", {"amount": 25})
    assert live.state["account:1"] == 145  # auto-updated


# --- 11. Disk persistence ---
def test_disk_persistence():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        s1 = EventStore(persist_path=path)
        s1.append("acc:1", "Opened", {"x": 1})
        s1.append("acc:1", "Deposited", {"amount": 50})

        s2 = EventStore(persist_path=path)
        assert len(s2.read_stream("acc:1")) == 2
        assert s2.stream_version("acc:1") == 2
    finally:
        os.unlink(path)


# --- 12. Global read ---
def test_global_read():
    store = EventStore()
    store.append("a", "X", {"v": 1})
    store.append("b", "Y", {"v": 2})
    store.append("a", "Z", {"v": 3})
    all_events = store.read_all()
    assert len(all_events) == 3
    assert [e.event_id for e in all_events] == [1, 2, 3]
    # from_position filters
    assert len(store.read_all(from_position=2)) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
