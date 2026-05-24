"""Verify the example usage from the spec works correctly."""
from event_store import *

store = EventStore()

store.append("account:1", "AccountOpened", {"owner": "Alice", "initial_balance": 0})
store.append("account:1", "MoneyDeposited", {"amount": 100})
store.append("account:1", "MoneyWithdrawn", {"amount": 30})
store.append("account:1", "MoneyDeposited", {"amount": 50})

events = store.read_stream("account:1")
assert len(events) == 4
assert events[0].event_type == "AccountOpened"

balance_projection = Projection("balances", store)

def on_opened(state, event):
    account = event.stream_id
    state[account] = event.data.get("initial_balance", 0)

def on_deposited(state, event):
    state[event.stream_id] = state.get(event.stream_id, 0) + event.data["amount"]

def on_withdrawn(state, event):
    state[event.stream_id] = state.get(event.stream_id, 0) - event.data["amount"]

balance_projection.when("AccountOpened", on_opened)
balance_projection.when("MoneyDeposited", on_deposited)
balance_projection.when("MoneyWithdrawn", on_withdrawn)

balance_projection.catch_up()
assert balance_projection.state["account:1"] == 120, f"Got {balance_projection.state}"

past_state = reconstruct_state(store, "account:1", {
    "AccountOpened": on_opened,
    "MoneyDeposited": on_deposited,
    "MoneyWithdrawn": on_withdrawn,
}, up_to=2)
assert past_state["account:1"] == 100, f"Got {past_state}"

store.append("account:1", "MoneyDeposited", {"amount": 10}, expected_version=4)

try:
    store.append("account:1", "MoneyDeposited", {"amount": 5}, expected_version=3)
    assert False
except ConcurrencyConflict:
    pass

balance_projection.catch_up()
balance_projection.save_snapshot()
new_projection = Projection("balances", store)
new_projection.when("AccountOpened", on_opened)
new_projection.when("MoneyDeposited", on_deposited)
new_projection.when("MoneyWithdrawn", on_withdrawn)
new_projection.load_snapshot()
assert new_projection.state["account:1"] == 130, f"Got {new_projection.state}"
assert new_projection.position == 5

store.append("account:1", "MoneyWithdrawn", {"amount": 20})
new_projection.catch_up()
assert new_projection.state["account:1"] == 110, f"Got {new_projection.state}"

# Test live projection
live = LiveProjection("live_balances", store)
live.when("AccountOpened", on_opened)
live.when("MoneyDeposited", on_deposited)
live.when("MoneyWithdrawn", on_withdrawn)
live.catch_up()

store.append("account:1", "MoneyDeposited", {"amount": 25})
assert live.state["account:1"] == 135, f"Live got {live.state}"

# Test disk persistence
import tempfile, os
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

# Test batch append
store2 = EventStore()
result = store2.append_batch("acc:2", [
    ("Opened", {"x": 1}),
    ("Deposited", {"amount": 100}),
    ("Withdrawn", {"amount": 30}),
])
assert len(result) == 3
assert store2.stream_version("acc:2") == 3

# Test stream isolation
store3 = EventStore()
store3.append("a", "X", {"v": 1})
store3.append("b", "Y", {"v": 2})
store3.append("a", "Z", {"v": 3})
assert len(store3.read_stream("a")) == 2
assert len(store3.read_stream("b")) == 1
assert len(store3.read_all()) == 3

# Test global read ordering
assert store3.read_all()[0].stream_id == "a"
assert store3.read_all()[1].stream_id == "b"
assert store3.read_all()[2].stream_id == "a"

# Test all_stream_ids
assert set(store3.all_stream_ids()) == {"a", "b"}

print("All assertions passed!")
