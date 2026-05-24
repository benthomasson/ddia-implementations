"""Tests for the unbundled database system."""

import os
import tempfile
import pytest
from unbundled_database import (
    WALEntry, CDCEvent, WriteAheadLog, StorageEngine, CDCStream,
    SecondaryIndex, MaterializedView, FullTextSearch, UnbundledDatabase,
)


# --- 1. Basic put/get/delete ---

class TestBasicPutGetDelete:
    def test_put_and_get(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        assert db.get("k1") == {"a": 1}

    def test_get_missing(self):
        db = UnbundledDatabase()
        assert db.get("nope") is None

    def test_delete(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        db.delete("k1")
        assert db.get("k1") is None

    def test_delete_nonexistent(self):
        db = UnbundledDatabase()
        assert db.delete("nope") is None

    def test_update(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        db.put("k1", {"a": 2})
        assert db.get("k1") == {"a": 2}


# --- 2. WAL tests ---

class TestWAL:
    def test_append_and_read(self):
        wal = WriteAheadLog()
        e1 = wal.append("PUT", "k1", {"v": 1})
        e2 = wal.append("PUT", "k2", {"v": 2})
        assert e1.lsn == 1
        assert e2.lsn == 2
        assert len(wal) == 2

    def test_read_from(self):
        wal = WriteAheadLog()
        wal.append("PUT", "k1", {"v": 1})
        wal.append("PUT", "k2", {"v": 2})
        wal.append("DELETE", "k1")
        entries = wal.read_from(2)
        assert len(entries) == 2
        assert entries[0].key == "k2"

    def test_latest_lsn(self):
        wal = WriteAheadLog()
        assert wal.latest_lsn == 0
        wal.append("PUT", "k1", {"v": 1})
        assert wal.latest_lsn == 1

    def test_earliest_lsn(self):
        wal = WriteAheadLog()
        assert wal.earliest_lsn == 0
        wal.append("PUT", "k1")
        assert wal.earliest_lsn == 1

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            wal1 = WriteAheadLog(persist_path=path)
            wal1.append("PUT", "k1", {"v": 1})
            wal1.append("PUT", "k2", {"v": 2})

            wal2 = WriteAheadLog(persist_path=path)
            assert len(wal2) == 2
            assert wal2.latest_lsn == 2
        finally:
            os.unlink(path)


# --- 3. CDC events ---

class TestCDCEvents:
    def test_insert_event(self):
        db = UnbundledDatabase()
        event = db.put("k1", {"a": 1})
        assert event.operation == "insert"
        assert event.new_value == {"a": 1}
        assert event.old_value is None

    def test_update_event(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        event = db.put("k1", {"a": 2})
        assert event.operation == "update"
        assert event.old_value == {"a": 1}
        assert event.new_value == {"a": 2}

    def test_delete_event(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        event = db.delete("k1")
        assert event.operation == "delete"
        assert event.old_value == {"a": 1}
        assert event.new_value is None


# --- 4. Secondary index ---

class TestSecondaryIndex:
    def test_query_after_insert(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"name": "Alice", "city": "NYC"})
        db.put("u2", {"name": "Bob", "city": "SF"})
        db.put("u3", {"name": "Carol", "city": "NYC"})
        db.flush()
        assert sorted(idx.query("city", "NYC")) == ["u1", "u3"]
        assert idx.query("city", "SF") == ["u2"]

    def test_query_after_update(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"city": "NYC"})
        db.flush()
        db.put("u1", {"city": "LA"})
        db.flush()
        assert idx.query("city", "NYC") == []
        assert idx.query("city", "LA") == ["u1"]

    def test_query_after_delete(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"city": "NYC"})
        db.flush()
        db.delete("u1")
        db.flush()
        assert idx.query("city", "NYC") == []


# --- 5. Materialized view ---

class TestMaterializedView:
    def test_count_aggregate(self):
        db = UnbundledDatabase()
        mv = MaterializedView("city_counts", "city", "count")
        db.add_derived_system(mv)
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "SF"})
        db.put("u3", {"city": "NYC"})
        db.flush()
        assert mv.query("NYC") == 2
        assert mv.query("SF") == 1
        assert mv.query("LA") == 0

    def test_count_after_update(self):
        db = UnbundledDatabase()
        mv = MaterializedView("city_counts", "city", "count")
        db.add_derived_system(mv)
        db.put("u1", {"city": "NYC"})
        db.flush()
        db.put("u1", {"city": "LA"})
        db.flush()
        assert mv.query("NYC") == 0
        assert mv.query("LA") == 1

    def test_count_after_delete(self):
        db = UnbundledDatabase()
        mv = MaterializedView("city_counts", "city", "count")
        db.add_derived_system(mv)
        db.put("u1", {"city": "SF"})
        db.flush()
        db.delete("u1")
        db.flush()
        assert mv.query("SF") == 0

    def test_list_aggregate(self):
        db = UnbundledDatabase()
        mv = MaterializedView("city_list", "city", "list")
        db.add_derived_system(mv)
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "NYC"})
        db.flush()
        assert sorted(mv.query("NYC")) == ["u1", "u2"]


# --- 6. Full-text search ---

class TestFullTextSearch:
    def test_single_term(self):
        db = UnbundledDatabase()
        fts = FullTextSearch("search", ["name", "bio"])
        db.add_derived_system(fts)
        db.put("u1", {"name": "Alice Smith", "bio": "Software engineer"})
        db.put("u2", {"name": "Bob Jones", "bio": "Data scientist"})
        db.flush()
        assert fts.search("engineer") == ["u1"]
        assert fts.search("data") == ["u2"]

    def test_multi_term(self):
        db = UnbundledDatabase()
        fts = FullTextSearch("search", ["bio"])
        db.add_derived_system(fts)
        db.put("u1", {"bio": "data scientist"})
        db.put("u2", {"bio": "data engineer"})
        db.flush()
        assert fts.search_all(["data", "scientist"]) == ["u1"]

    def test_case_insensitive(self):
        db = UnbundledDatabase()
        fts = FullTextSearch("search", ["name"])
        db.add_derived_system(fts)
        db.put("u1", {"name": "Alice"})
        db.flush()
        assert fts.search("alice") == ["u1"]
        assert fts.search("ALICE") == ["u1"]

    def test_search_after_delete(self):
        db = UnbundledDatabase()
        fts = FullTextSearch("search", ["name"])
        db.add_derived_system(fts)
        db.put("u1", {"name": "Alice"})
        db.flush()
        db.delete("u1")
        db.flush()
        assert fts.search("alice") == []


# --- 7. Catch-up ---

class TestCatchUp:
    def test_new_system_catches_up(self):
        db = UnbundledDatabase()
        db.put("u1", {"name": "Alice", "city": "NYC"})
        db.put("u2", {"name": "Bob", "city": "SF"})
        db.flush()

        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx, catch_up=True)
        assert sorted(idx.query("city", "NYC")) == ["u1"]
        assert sorted(idx.query("city", "SF")) == ["u2"]

    def test_catch_up_reflects_deletes(self):
        db = UnbundledDatabase()
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "SF"})
        db.delete("u1")

        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx, catch_up=True)
        assert idx.query("city", "NYC") == []
        assert idx.query("city", "SF") == ["u2"]


# --- 8. Rebuild derived system ---

class TestRebuild:
    def test_rebuild_matches_live(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "SF"})
        db.put("u1", {"city": "LA"})
        db.delete("u2")
        db.flush()

        live_state = idx.get_state()
        db.rebuild_system("by_city")
        rebuilt_state = idx.get_state()
        assert live_state == rebuilt_state


# --- 9. Storage engine rebuild from WAL ---

class TestStorageRebuild:
    def test_rebuild_from_wal(self):
        db = UnbundledDatabase()
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "SF"})
        db.put("u1", {"city": "LA"})
        db.delete("u2")

        db.storage.rebuild(db.wal)
        assert db.get("u1") == {"city": "LA"}
        assert db.get("u2") is None


# --- 10. Lag monitoring ---

class TestLag:
    def test_lag_increases_after_write(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("idx", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"city": "NYC"})
        lag = db.get_lag()
        assert lag["idx"] > 0

    def test_lag_zero_after_flush(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("idx", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"city": "NYC"})
        db.flush()
        lag = db.get_lag()
        assert lag["idx"] == 0


# --- 11. Pipeline introspection ---

class TestPipelineIntrospection:
    def test_pipeline_state(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        db.add_derived_system(idx)
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "SF"})
        db.flush()

        state = db.get_pipeline_state()
        assert state["wal_size"] == 2
        assert state["storage_records"] == 2
        assert state["cdc_events"] == 2
        assert len(state["derived_systems"]) == 1
        assert state["derived_systems"][0]["name"] == "by_city"
        assert state["derived_systems"][0]["lag"] == 0


# --- 12. CDC old_value ---

class TestCDCOldValue:
    def test_insert_has_no_old_value(self):
        db = UnbundledDatabase()
        event = db.put("k1", {"a": 1})
        assert event.old_value is None

    def test_update_has_old_value(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        event = db.put("k1", {"a": 2})
        assert event.old_value == {"a": 1}

    def test_delete_has_old_value(self):
        db = UnbundledDatabase()
        db.put("k1", {"a": 1})
        event = db.delete("k1")
        assert event.old_value == {"a": 1}


# --- 13. Delete cascade ---

class TestDeleteCascade:
    def test_derived_systems_updated_on_delete(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        mv = MaterializedView("counts", "city", "count")
        fts = FullTextSearch("search", ["name"])
        db.add_derived_system(idx)
        db.add_derived_system(mv)
        db.add_derived_system(fts)

        db.put("u1", {"name": "Alice", "city": "NYC"})
        db.flush()
        assert idx.query("city", "NYC") == ["u1"]
        assert mv.query("NYC") == 1
        assert fts.search("alice") == ["u1"]

        db.delete("u1")
        db.flush()
        assert idx.query("city", "NYC") == []
        assert mv.query("NYC") == 0
        assert fts.search("alice") == []


# --- 14. Multiple independent consumers ---

class TestMultipleConsumers:
    def test_independent_processing(self):
        db = UnbundledDatabase()
        idx1 = SecondaryIndex("by_city", ["city"])
        idx2 = SecondaryIndex("by_name", ["name"])
        db.add_derived_system(idx1)
        db.add_derived_system(idx2)

        db.put("u1", {"name": "Alice", "city": "NYC"})
        db.flush()

        assert idx1.query("city", "NYC") == ["u1"]
        assert idx2.query("name", "Alice") == ["u1"]

        # Unsubscribe one, the other still works
        db.cdc.unsubscribe("by_city")
        db.put("u2", {"name": "Bob", "city": "SF"})
        db.flush()
        # by_city didn't get the update (unsubscribed)
        assert idx1.query("city", "SF") == []
        assert idx2.query("name", "Bob") == ["u2"]


# --- 15. Log truncation ---

class TestLogTruncation:
    def test_truncate(self):
        wal = WriteAheadLog()
        wal.append("PUT", "k1", {"v": 1})
        wal.append("PUT", "k2", {"v": 2})
        wal.append("PUT", "k3", {"v": 3})
        removed = wal.truncate_before(3)
        assert removed == 2
        assert len(wal) == 1
        assert wal.earliest_lsn == 3

    def test_truncate_all(self):
        wal = WriteAheadLog()
        wal.append("PUT", "k1")
        wal.append("PUT", "k2")
        removed = wal.truncate_before(10)
        assert removed == 2
        assert len(wal) == 0


# --- 16. End-to-end flow ---

class TestEndToEnd:
    def test_full_pipeline(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_city", ["city"])
        mv = MaterializedView("city_counts", "city", "count")
        fts = FullTextSearch("name_search", ["name", "bio"])
        db.add_derived_system(idx)
        db.add_derived_system(mv)
        db.add_derived_system(fts)

        # Write data
        db.put("user:1", {"name": "Alice Smith", "city": "NYC", "bio": "Software engineer"})
        db.put("user:2", {"name": "Bob Jones", "city": "SF", "bio": "Data scientist"})
        db.put("user:3", {"name": "Carol White", "city": "NYC", "bio": "Product manager"})
        db.flush()

        # Verify WAL
        assert db.wal.latest_lsn == 3
        assert len(db.wal) == 3

        # Verify storage
        assert db.storage.record_count == 3
        assert db.get("user:1")["name"] == "Alice Smith"

        # Verify CDC events
        assert len(db.cdc.events) == 3

        # Verify secondary index
        nyc_users = db.query_index("by_city", "city", "NYC")
        assert len(nyc_users) == 2

        # Verify materialized view
        assert mv.query("NYC") == 2
        assert mv.query("SF") == 1

        # Verify full-text search
        assert fts.search("engineer") == ["user:1"]
        assert fts.search_all(["data", "scientist"]) == ["user:2"]

        # Update
        event = db.put("user:1", {"name": "Alice Smith", "city": "LA", "bio": "Tech lead"})
        assert event.operation == "update"
        assert event.old_value["city"] == "NYC"
        assert event.new_value["city"] == "LA"
        db.flush()

        nyc_users = db.query_index("by_city", "city", "NYC")
        assert len(nyc_users) == 1
        la_users = db.query_index("by_city", "city", "LA")
        assert len(la_users) == 1
        assert mv.query("NYC") == 1
        assert mv.query("LA") == 1

        # Delete
        db.delete("user:2")
        db.flush()
        assert db.get("user:2") is None
        assert mv.query("SF") == 0

        # Catch-up
        new_idx = SecondaryIndex("by_name_first_letter", ["name"])
        db.add_derived_system(new_idx, catch_up=True)
        assert len(new_idx.query("name", "Alice Smith")) == 1

        # Rebuild
        count = db.rebuild_system("by_city")
        assert count >= 3

        # Lag
        db.put("user:4", {"name": "Dave Brown", "city": "NYC", "bio": "Designer"})
        lag = db.get_lag()
        assert any(v > 0 for v in lag.values())
        db.flush()
        lag = db.get_lag()
        assert all(v == 0 for v in lag.values())

        # Pipeline introspection
        state = db.get_pipeline_state()
        assert state["storage_records"] == 3
        assert len(state["derived_systems"]) == 4

        # Storage rebuild
        db.storage.rebuild(db.wal)
        assert db.get("user:1")["city"] == "LA"


# --- Performance test ---

class TestPerformance:
    def test_10k_records(self):
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_category", ["category"])
        mv = MaterializedView("cat_counts", "category", "count")
        db.add_derived_system(idx)
        db.add_derived_system(mv)

        categories = ["A", "B", "C", "D", "E"]
        for i in range(10_000):
            cat = categories[i % len(categories)]
            db.put(f"item:{i}", {"category": cat, "value": i})

        db.flush()
        assert mv.query("A") == 2000
        assert len(idx.query("category", "B")) == 2000
        assert db.storage.record_count == 10_000


# --- Scan test ---

class TestScan:
    def test_scan_prefix(self):
        db = UnbundledDatabase()
        db.put("user:1", {"name": "Alice"})
        db.put("user:2", {"name": "Bob"})
        db.put("order:1", {"total": 100})
        users = db.storage.scan("user:")
        assert len(users) == 2
        assert "user:1" in users


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
