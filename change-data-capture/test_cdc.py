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


@pytest.fixture
def populated_db(db):
    db.insert("users", {"id": 1, "name": "Alice", "email": "alice@example.com", "city": "NYC"})
    db.insert("users", {"id": 2, "name": "Bob", "email": "bob@example.com", "city": "LA"})
    return db


# --- 1. Insert/Update/Delete generate correct ChangeEvents ---

class TestChangeEvents:
    def test_insert_event(self, db):
        db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
        events = db.cdc_log.read_from(0)
        assert len(events) == 1
        e = events[0]
        assert e.operation == Operation.INSERT
        assert e.table == "users"
        assert e.key == 1
        assert e.sequence_number == 0

    def test_update_event(self, populated_db):
        populated_db.update("users", 1, {"city": "SF"})
        e = populated_db.cdc_log.read_from(2)[0]
        assert e.operation == Operation.UPDATE
        assert e.key == 1

    def test_delete_event(self, populated_db):
        populated_db.delete("users", 2)
        e = populated_db.cdc_log.read_from(2)[0]
        assert e.operation == Operation.DELETE
        assert e.key == 2


# --- 2. Before/after states ---

class TestBeforeAfter:
    def test_insert_before_after(self, db):
        db.insert("users", {"id": 1, "name": "Alice", "email": "a@b.com", "city": "NYC"})
        e = db.cdc_log.read_from(0)[0]
        assert e.before is None
        assert e.after["name"] == "Alice"

    def test_update_before_after(self, populated_db):
        populated_db.update("users", 1, {"city": "SF"})
        e = populated_db.cdc_log.read_from(2)[0]
        assert e.before["city"] == "NYC"
        assert e.after["city"] == "SF"
        # Other fields preserved
        assert e.after["name"] == "Alice"
        assert e.before["name"] == "Alice"

    def test_delete_before_after(self, populated_db):
        populated_db.delete("users", 2)
        e = populated_db.cdc_log.read_from(2)[0]
        assert e.before["name"] == "Bob"
        assert e.after is None


# --- 3. Consumer processes events in order and tracks position ---

class TestConsumer:
    def test_poll_processes_all(self, populated_db):
        log = populated_db.cdc_log
        events_seen = []
        c = CDCConsumer("test", log)
        c.on_all(lambda e: events_seen.append(e))
        count = c.poll()
        assert count == 2
        assert len(events_seen) == 2
        assert events_seen[0].key == 1
        assert events_seen[1].key == 2

    def test_poll_incremental(self, populated_db):
        log = populated_db.cdc_log
        c = CDCConsumer("test", log)
        seen = []
        c.on_all(lambda e: seen.append(e))
        c.poll()
        assert len(seen) == 2

        populated_db.insert("users", {"id": 3, "name": "Charlie", "email": "c@d.com", "city": "LA"})
        c.poll()
        assert len(seen) == 3

    def test_position_tracking(self, populated_db):
        c = CDCConsumer("test", populated_db.cdc_log)
        c.on_all(lambda e: None)
        assert c.position == 0
        c.poll()
        assert c.position == 2

    def test_filtered_handler(self, db):
        db.create_table("orders", ["id", "amount"], primary_key="id")
        db.insert("users", {"id": 1, "name": "A", "email": "a@b", "city": "X"})
        db.insert("orders", {"id": 10, "amount": 99})

        user_events = []
        c = CDCConsumer("test", db.cdc_log)
        c.on("users", None, lambda e: user_events.append(e))
        c.poll()
        assert len(user_events) == 1
        assert user_events[0].table == "users"

    def test_operation_filter(self, populated_db):
        populated_db.update("users", 1, {"city": "SF"})
        updates = []
        c = CDCConsumer("test", populated_db.cdc_log)
        c.on("users", Operation.UPDATE, lambda e: updates.append(e))
        c.poll()
        assert len(updates) == 1


# --- 4. Multiple independent consumers ---

class TestMultipleConsumers:
    def test_independent_consumers(self, populated_db):
        log = populated_db.cdc_log
        seen_a, seen_b = [], []
        a = CDCConsumer("a", log)
        b = CDCConsumer("b", log)
        a.on_all(lambda e: seen_a.append(e))
        b.on_all(lambda e: seen_b.append(e))

        a.poll()
        assert len(seen_a) == 2
        assert len(seen_b) == 0  # b hasn't polled yet

        populated_db.insert("users", {"id": 3, "name": "C", "email": "c@d", "city": "X"})
        b.poll()
        assert len(seen_b) == 3  # b sees all 3
        a.poll()
        assert len(seen_a) == 3  # a sees the new one


# --- 5. Consumer seek ---

class TestSeek:
    def test_seek_and_replay(self, populated_db):
        log = populated_db.cdc_log
        seen = []
        c = CDCConsumer("test", log)
        c.on_all(lambda e: seen.append(e))
        c.poll()
        assert len(seen) == 2

        c.seek(0)
        c.poll()
        assert len(seen) == 4  # replayed all 2 again


# --- 6. MaterializedView mirrors source ---

class TestMaterializedView:
    def test_basic_replication(self, populated_db):
        mv = MaterializedView("copy", "users", populated_db.cdc_log)
        mv.refresh()
        assert mv.get(1)["name"] == "Alice"
        assert mv.get(2)["name"] == "Bob"

    def test_updates_reflected(self, populated_db):
        mv = MaterializedView("copy", "users", populated_db.cdc_log)
        mv.refresh()
        populated_db.update("users", 1, {"city": "SF"})
        mv.refresh()
        assert mv.get(1)["city"] == "SF"

    def test_deletes_reflected(self, populated_db):
        mv = MaterializedView("copy", "users", populated_db.cdc_log)
        mv.refresh()
        populated_db.delete("users", 2)
        mv.refresh()
        assert mv.get(2) is None

    def test_scan(self, populated_db):
        mv = MaterializedView("copy", "users", populated_db.cdc_log)
        mv.refresh()
        rows = mv.scan()
        assert len(rows) == 2


# --- 7. MaterializedView with transform ---

class TestMaterializedViewTransform:
    def test_filter_transform(self, populated_db):
        # Only keep NYC users, and only name+city
        def xform(row):
            if row["city"] != "NYC":
                return None
            return {"name": row["name"], "city": row["city"]}

        mv = MaterializedView("nyc_users", "users", populated_db.cdc_log, transform=xform)
        mv.refresh()
        assert mv.get(1) is not None
        assert mv.get(1)["name"] == "Alice"
        assert mv.get(2) is None  # Bob is in LA

    def test_transform_on_update(self, populated_db):
        def xform(row):
            if row["city"] != "NYC":
                return None
            return row

        mv = MaterializedView("nyc", "users", populated_db.cdc_log, transform=xform)
        mv.refresh()
        assert mv.get(1) is not None

        populated_db.update("users", 1, {"city": "LA"})
        mv.refresh()
        assert mv.get(1) is None  # Alice left NYC


# --- 8. SearchIndex ---

class TestSearchIndex:
    def test_basic_search(self, populated_db):
        si = SearchIndex("idx", "users", populated_db.cdc_log, ["name", "email"])
        si.refresh()
        results = si.search("alice")
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_search_after_update(self, populated_db):
        si = SearchIndex("idx", "users", populated_db.cdc_log, ["name"])
        si.refresh()
        populated_db.update("users", 1, {"name": "Alicia"})
        si.refresh()
        assert len(si.search("alice")) == 0
        assert len(si.search("alicia")) == 1

    def test_search_after_delete(self, populated_db):
        si = SearchIndex("idx", "users", populated_db.cdc_log, ["name"])
        si.refresh()
        populated_db.delete("users", 1)
        si.refresh()
        assert len(si.search("alice")) == 0

    def test_search_no_results(self, populated_db):
        si = SearchIndex("idx", "users", populated_db.cdc_log, ["name"])
        si.refresh()
        assert len(si.search("nonexistent")) == 0


# --- 9. Snapshotting ---

class TestSnapshot:
    def test_snapshot_current_state(self, populated_db):
        populated_db.delete("users", 2)
        events, pos = create_snapshot(populated_db, "users")
        assert len(events) == 1  # only Alice
        assert events[0].after["name"] == "Alice"
        assert events[0].operation == Operation.INSERT
        assert events[0].sequence_number == -1
        assert pos == populated_db.cdc_log.current_position


# --- 10. Log compaction ---

class TestCompaction:
    def test_compaction_removes_intermediate(self, db):
        db.insert("users", {"id": 1, "name": "A", "email": "a@b", "city": "X"})
        db.update("users", 1, {"city": "Y"})
        db.update("users", 1, {"city": "Z"})
        assert len(db.cdc_log.read_from(0)) == 3
        removed = db.cdc_log.compact()
        assert removed == 2
        events = db.cdc_log.read_from(0)
        assert len(events) == 1
        assert events[0].after["city"] == "Z"

    def test_compaction_preserves_different_keys(self, populated_db):
        populated_db.cdc_log.compact()
        events = populated_db.cdc_log.read_from(0)
        assert len(events) == 2  # one per user

    def test_reconstruct_after_compaction(self, db):
        db.insert("users", {"id": 1, "name": "A", "email": "a@b", "city": "X"})
        db.update("users", 1, {"city": "Y"})
        db.delete("users", 1)
        db.cdc_log.compact()
        events = db.cdc_log.read_from(0)
        assert len(events) == 1
        assert events[0].operation == Operation.DELETE


# --- 11. Sequence numbers strictly increasing ---

class TestSequenceNumbers:
    def test_strictly_increasing(self, db):
        for i in range(10):
            db.insert("users", {"id": i, "name": f"U{i}", "email": f"{i}@x", "city": "X"})
        events = db.cdc_log.read_from(0)
        for i in range(1, len(events)):
            assert events[i].sequence_number > events[i - 1].sequence_number


# --- 12. Error cases ---

class TestErrors:
    def test_duplicate_insert(self, populated_db):
        with pytest.raises(ValueError):
            populated_db.insert("users", {"id": 1, "name": "Dup", "email": "x", "city": "X"})

    def test_update_nonexistent(self, db):
        with pytest.raises(KeyError):
            db.update("users", 999, {"city": "X"})

    def test_delete_nonexistent(self, db):
        with pytest.raises(KeyError):
            db.delete("users", 999)


# --- Example from spec ---

class TestSpecExample:
    def test_full_example(self):
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

        audit_entries = []
        auditor = CDCConsumer("audit", log)
        auditor.on_all(lambda e: audit_entries.append(
            f"{e.operation.value} on {e.table} key={e.key}"
        ))
        auditor.poll()
        assert len(audit_entries) == 5

        users_replica = MaterializedView("users_copy", "users", log)
        users_replica.refresh()
        assert users_replica.get(1)["city"] == "SF"
        assert users_replica.get(2) is None

        search = SearchIndex("user_search", "users", log, ["name", "email"])
        search.refresh()
        results = search.search("alice")
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

        snapshot_events, snapshot_pos = create_snapshot(db, "users")
        assert len(snapshot_events) == 1
        assert snapshot_events[0].after["name"] == "Alice"

        db.insert("users", {"id": 3, "name": "Charlie", "email": "c@example.com", "city": "LA"})
        db.update("users", 3, {"city": "Denver"})
        db.update("users", 3, {"city": "Boston"})
        removed = log.compact()
        assert removed > 0
