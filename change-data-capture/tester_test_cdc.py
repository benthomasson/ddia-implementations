"""Tests for the CDC system."""


import pytest
from cdc import (
    CDCDatabase, CDCLog, CDCConsumer, MaterializedView, SearchIndex,
    Operation, ChangeEvent, create_snapshot,
)


@pytest.fixture
def db():
    d = CDCDatabase()
    d.create_table("users", ["id", "name", "email", "city"], primary_key="id")
    return d


def test_full_spec_example():
    """Run the example from the spec verbatim."""
    db = CDCDatabase()
    db.create_table("users", ["id", "name", "email", "city"], primary_key="id")
    db.create_table("orders", ["id", "user_id", "product", "amount"], primary_key="id")

    db.insert("users", {"id": 1, "name": "Alice", "email": "alice@example.com", "city": "NYC"})
    db.insert("users", {"id": 2, "name": "Bob", "email": "bob@example.com", "city": "LA"})
    db.insert("orders", {"id": 101, "user_id": 1, "product": "Widget", "amount": 25.0})

    log = db.cdc_log
    events = log.read_from(0)
    assert len(events) == 3
    assert events[0].operation == Operation.INSERT
    assert events[0].table == "users"
    assert events[0].after["name"] == "Alice"

    db.update("users", 1, {"city": "SF"})
    update_event = log.read_from(3)[0]
    assert update_event.operation == Operation.UPDATE
    assert update_event.before["city"] == "NYC"
    assert update_event.after["city"] == "SF"

    db.delete("users", 2)
    delete_event = log.read_from(4)[0]
    assert delete_event.operation == Operation.DELETE
    assert delete_event.before["name"] == "Bob"
    assert delete_event.after is None

    # Consumer: audit log
    audit_entries = []
    auditor = CDCConsumer("audit", log)
    auditor.on_all(lambda e: audit_entries.append(
        f"{e.operation.value} on {e.table} key={e.key}"
    ))
    auditor.poll()
    assert len(audit_entries) == 5

    # Materialized view
    users_replica = MaterializedView("users_copy", "users", log)
    users_replica.refresh()
    assert users_replica.get(1)["city"] == "SF"
    assert users_replica.get(2) is None  # was deleted

    # Search index
    search = SearchIndex("user_search", "users", log, ["name", "email"])
    search.refresh()
    results = search.search("alice")
    assert len(results) == 1
    assert results[0]["name"] == "Alice"

    # Snapshot
    snapshot_events, snapshot_pos = create_snapshot(db, "users")
    assert len(snapshot_events) == 1  # only Alice remains
    assert snapshot_events[0].after["name"] == "Alice"

    # Compaction
    db.insert("users", {"id": 3, "name": "Charlie", "email": "c@example.com", "city": "LA"})
    db.update("users", 3, {"city": "Denver"})
    db.update("users", 3, {"city": "Boston"})
    removed = log.compact()
    assert removed > 0


def test_before_after_states(db):
    """Test before/after for insert, update, delete."""
    db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
    e = db.cdc_log.read_from(0)[0]
    assert e.before is None
    assert e.after == {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"}

    db.update("users", 1, {"city": "SF"})
    e = db.cdc_log.read_from(1)[0]
    assert e.before["city"] == "NYC"
    assert e.after["city"] == "SF"
    assert e.after["name"] == "Alice"

    db.delete("users", 1)
    e = db.cdc_log.read_from(2)[0]
    assert e.before["name"] == "Alice"
    assert e.before["city"] == "SF"
    assert e.after is None


def test_multiple_independent_consumers(db):
    """Two consumers process same log independently."""
    db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
    db.insert("users", {"id": 2, "name": "Bob", "email": "b@b.com", "city": "LA"})

    log = db.cdc_log
    seen_a, seen_b = [], []
    a = CDCConsumer("a", log)
    b = CDCConsumer("b", log)
    a.on_all(lambda e: seen_a.append(e))
    b.on_all(lambda e: seen_b.append(e))

    a.poll()
    assert len(seen_a) == 2
    assert len(seen_b) == 0  # b hasn't polled yet

    db.insert("users", {"id": 3, "name": "Charlie", "email": "c@b.com", "city": "SF"})
    b.poll()
    assert len(seen_b) == 3  # b gets all 3
    a.poll()
    assert len(seen_a) == 3  # a gets only the new one


def test_consumer_seek(db):
    """Seek resets position for replay."""
    db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
    db.insert("users", {"id": 2, "name": "Bob", "email": "b@b.com", "city": "LA"})

    log = db.cdc_log
    seen = []
    c = CDCConsumer("c", log)
    c.on_all(lambda e: seen.append(e))
    c.poll()
    assert len(seen) == 2

    c.seek(0)
    c.poll()
    assert len(seen) == 4  # replayed both events


def test_materialized_view_with_transform(db):
    """MaterializedView filters/transforms rows."""
    db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
    db.insert("users", {"id": 2, "name": "Bob", "email": "b@b.com", "city": "LA"})

    # Only keep NYC users, project name only
    def nyc_only(row):
        if row["city"] != "NYC":
            return None
        return {"id": row["id"], "name": row["name"]}

    view = MaterializedView("nyc", "users", db.cdc_log, transform=nyc_only)
    view.refresh()
    assert view.get(1) == {"id": 1, "name": "Alice"}
    assert view.get(2) is None  # filtered out


def test_search_index_update_delete(db):
    """Search index updates when rows change or are deleted."""
    db.insert("users", {"id": 1, "name": "Alice Smith", "email": "a@b.com", "city": "NYC"})
    search = SearchIndex("idx", "users", db.cdc_log, ["name"])
    search.refresh()
    assert len(search.search("alice")) == 1
    assert len(search.search("smith")) == 1

    db.update("users", 1, {"name": "Alice Jones"})
    search.refresh()
    assert len(search.search("jones")) == 1
    assert len(search.search("smith")) == 0  # old token removed

    db.delete("users", 1)
    search.refresh()
    assert len(search.search("alice")) == 0
    assert len(search.search("jones")) == 0


def test_error_cases(db):
    """Duplicate insert, update/delete non-existent rows."""
    db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
    with pytest.raises(ValueError):
        db.insert("users", {"id": 1, "name": "Dup", "email": "d@b.com", "city": "X"})
    with pytest.raises(KeyError):
        db.update("users", 999, {"city": "X"})
    with pytest.raises(KeyError):
        db.delete("users", 999)


def test_compaction_preserves_state(db):
    """After compaction, log still reconstructs current state."""
    db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
    db.update("users", 1, {"city": "SF"})
    db.update("users", 1, {"city": "Denver"})
    db.insert("users", {"id": 2, "name": "Bob", "email": "b@b.com", "city": "LA"})

    log = db.cdc_log
    assert len(log.read_from(0)) == 4
    removed = log.compact()
    assert removed == 2  # two intermediate events for user 1
    events = log.read_from(0)
    assert len(events) == 2  # one per key

    # Rebuild from compacted log
    view = MaterializedView("rebuilt", "users", log)
    view.refresh()
    assert view.get(1)["city"] == "Denver"
    assert view.get(2)["name"] == "Bob"


def test_sequence_numbers_strictly_increasing(db):
    """Sequence numbers increase monotonically."""
    db.insert("users", {"id": 1, "name": "A", "email": "a@b.com", "city": "X"})
    db.insert("users", {"id": 2, "name": "B", "email": "b@b.com", "city": "Y"})
    db.update("users", 1, {"city": "Z"})
    db.delete("users", 2)

    events = db.cdc_log.read_from(0)
    for i in range(1, len(events)):
        assert events[i].sequence_number > events[i - 1].sequence_number
