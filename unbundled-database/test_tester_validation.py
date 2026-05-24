"""Tester-stage validation: edge cases and spec example verification."""

import os
import tempfile
import pytest
from unbundled_database import (
    WALEntry, CDCEvent, WriteAheadLog, StorageEngine, CDCStream,
    SecondaryIndex, MaterializedView, FullTextSearch, UnbundledDatabase,
)


class TestSpecExample:
    """Run the exact example from the spec's Example Usage section."""

    def test_full_spec_example(self):
        db = UnbundledDatabase()

        idx = SecondaryIndex("by_city", indexed_fields=["city"])
        mv = MaterializedView("city_counts", group_by_field="city", aggregate="count")
        fts = FullTextSearch("name_search", text_fields=["name", "bio"])

        db.add_derived_system(idx)
        db.add_derived_system(mv)
        db.add_derived_system(fts)

        db.put("user:1", {"name": "Alice Smith", "city": "NYC", "bio": "Software engineer"})
        db.put("user:2", {"name": "Bob Jones", "city": "SF", "bio": "Data scientist"})
        db.put("user:3", {"name": "Carol White", "city": "NYC", "bio": "Product manager"})
        db.flush()

        nyc_users = db.query_index("by_city", "city", "NYC")
        assert len(nyc_users) == 2
        assert any(u["name"] == "Alice Smith" for u in nyc_users)

        assert mv.query("NYC") == 2
        assert mv.query("SF") == 1

        results = fts.search("engineer")
        assert "user:1" in results
        results = fts.search_all(["data", "scientist"])
        assert "user:2" in results

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

        db.delete("user:2")
        db.flush()
        assert db.get("user:2") is None
        assert mv.query("SF") == 0

        new_idx = SecondaryIndex("by_name_first_letter", indexed_fields=["name"])
        db.add_derived_system(new_idx, catch_up=True)
        assert len(new_idx.query("name", "Alice Smith")) == 1

        count = db.rebuild_system("by_city")
        assert count >= 3

        db.put("user:4", {"name": "Dave Brown", "city": "NYC", "bio": "Designer"})
        lag = db.get_lag()
        assert any(v > 0 for v in lag.values())
        db.flush()
        lag = db.get_lag()
        assert all(v == 0 for v in lag.values())

        state = db.get_pipeline_state()
        assert state["storage_records"] == 3
        assert len(state["derived_systems"]) == 4

        db.storage.rebuild(db.wal)
        assert db.get("user:1")["city"] == "LA"


class TestEdgeCases:
    """Edge cases from reviewer notes and spec constraints."""

    def test_rebuild_after_catch_up_matches_live(self):
        """Catch-up via snapshot_and_stream should produce same state as rebuild."""
        db = UnbundledDatabase()
        db.put("u1", {"city": "NYC"})
        db.put("u2", {"city": "SF"})
        db.put("u1", {"city": "LA"})  # update
        db.delete("u2")

        # Catch-up system
        idx_catchup = SecondaryIndex("catchup", ["city"])
        db.add_derived_system(idx_catchup, catch_up=True)
        catchup_state = idx_catchup.get_state()

        # Rebuild system (replays full CDC event log)
        idx_rebuild = SecondaryIndex("rebuild", ["city"])
        db.add_derived_system(idx_rebuild, catch_up=False)
        idx_rebuild.rebuild(db.cdc.events)
        rebuild_state = idx_rebuild.get_state()

        # Catch-up uses storage snapshot (current state), rebuild replays events
        # Both should show only u1 -> LA
        assert catchup_state == rebuild_state

    def test_lsn_sequential_no_gaps(self):
        """LSNs start at 1 and are sequential with no gaps."""
        db = UnbundledDatabase()
        events = []
        for i in range(5):
            events.append(db.put(f"k{i}", {"v": i}))
        for i, e in enumerate(events):
            assert e.lsn == i + 1

    def test_nested_dict_values(self):
        """System handles records with nested dict values."""
        db = UnbundledDatabase()
        idx = SecondaryIndex("by_status", ["status"])
        db.add_derived_system(idx)

        db.put("u1", {"name": "Alice", "status": "active", "meta": {"role": "admin"}})
        db.flush()
        assert idx.query("status", "active") == ["u1"]
        assert db.get("u1")["meta"]["role"] == "admin"

    def test_empty_search_all(self):
        """search_all with empty terms returns empty list."""
        fts = FullTextSearch("search", ["name"])
        assert fts.search_all([]) == []

    def test_storage_engine_rebuild_clears_old_state(self):
        """Rebuild clears state completely before replaying."""
        db = UnbundledDatabase()
        db.put("u1", {"v": 1})
        db.put("u2", {"v": 2})
        assert db.storage.record_count == 2

        # Manually inject extra state that shouldn't survive rebuild
        db.storage._data["phantom"] = {"v": 999}
        assert db.storage.record_count == 3

        db.storage.rebuild(db.wal)
        assert db.storage.record_count == 2
        assert db.get("phantom") is None

    def test_add_derived_system_no_catch_up(self):
        """Adding a system with catch_up=False starts empty."""
        db = UnbundledDatabase()
        db.put("u1", {"city": "NYC"})

        idx = SecondaryIndex("lazy", ["city"])
        db.add_derived_system(idx, catch_up=False)
        # Should have no data yet (no catch-up, no flush of existing)
        assert idx.query("city", "NYC") == []

        # But future writes (after flush) should work
        db.put("u2", {"city": "SF"})
        db.flush()
        assert idx.query("city", "SF") == ["u2"]

    def test_wal_persistence_roundtrip(self):
        """WAL persisted to disk can fully reconstruct storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = UnbundledDatabase(persist_dir=tmpdir)
            db1.put("k1", {"a": 1})
            db1.put("k2", {"b": 2})
            db1.delete("k1")

            # Create new database from same directory
            db2 = UnbundledDatabase(persist_dir=tmpdir)
            # WAL was restored; rebuild storage from it
            db2.storage.rebuild(db2.wal)
            assert db2.get("k1") is None
            assert db2.get("k2") == {"b": 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
