"""Tests for secondary index partitioning implementation."""
import pytest

from secondary_index_partitioning import (
    Document, Partition, OperationResult,
    DocumentPartitionedDB, TermPartitionedDB, compare_strategies,
)


# --- 1. Basic CRUD ---
class TestCRUD:
    def test_put_get_delete(self):
        for DB in [DocumentPartitionedDB, TermPartitionedDB]:
            db = DB(4, ["color"])
            db.put("p1", {"name": "Widget", "color": "red"})
            r = db.get("p1")
            assert r.data["name"] == "Widget"
            assert r.data["color"] == "red"
            assert r.partitions_touched == 1
            db.delete("p1")
            assert db.get("p1").data is None

    def test_get_nonexistent(self):
        db = DocumentPartitionedDB(4, ["color"])
        assert db.get("missing").data is None

    def test_update_replaces_data(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p1", {"color": "blue"})
        assert db.get("p1").data["color"] == "blue"


# --- 2. Document-partitioned index + scatter/gather ---
class TestDocPartitioned:
    def test_scatter_gather_touches_all_partitions(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p2", {"color": "red"})
        db.put("p3", {"color": "blue"})
        r = db.query_by_field("color", "red")
        pks = {pk for pk, _ in r.data}
        assert pks == {"p1", "p2"}
        assert r.partitions_touched == 4  # always all partitions

    def test_write_touches_one_partition(self):
        db = DocumentPartitionedDB(4, ["color"])
        r = db.put("p1", {"color": "red"})
        assert r.partitions_touched == 1

    def test_update_removes_old_index_entry(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p1", {"color": "blue"})
        assert len(db.query_by_field("color", "red").data) == 0
        assert len(db.query_by_field("color", "blue").data) == 1

    def test_delete_cleans_index(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.delete("p1")
        assert len(db.query_by_field("color", "red").data) == 0


# --- 3. Term-partitioned index ---
class TestTermPartitioned:
    def test_query_touches_fewer_partitions(self):
        db = TermPartitionedDB(4, ["color"])
        for i in range(20):
            db.put(f"p{i}", {"color": "red"})
        r = db.query_by_field("color", "red")
        assert len(r.data) == 20
        assert r.partitions_touched <= 4  # index partition + data partitions

    def test_write_touches_multiple_partitions(self):
        """With 2 indexed fields, a write may touch >1 partition for index updates."""
        db = TermPartitionedDB(4, ["color", "size"])
        r = db.put("p1", {"color": "red", "size": "large"})
        assert r.partitions_touched >= 1

    def test_query_nonexistent_value(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        r = db.query_by_field("color", "purple")
        assert len(r.data) == 0

    def test_update_cleans_old_global_index(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p1", {"color": "blue"})
        assert len(db.query_by_field("color", "red").data) == 0
        assert len(db.query_by_field("color", "blue").data) == 1

    def test_delete_cleans_global_index(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.delete("p1")
        assert len(db.query_by_field("color", "red").data) == 0


# --- 4. Async index mode ---
class TestAsyncIndex:
    def test_stale_before_flush(self):
        db = TermPartitionedDB(4, ["color"], async_index=True)
        db.put("p1", {"color": "red"})
        assert len(db.query_by_field("color", "red").data) == 0  # stale

    def test_correct_after_flush(self):
        db = TermPartitionedDB(4, ["color"], async_index=True)
        db.put("p1", {"color": "red"})
        count = db.flush_index()
        assert count >= 1
        assert len(db.query_by_field("color", "red").data) == 1


# --- 5. Range queries ---
class TestRangeQuery:
    def test_range_query_filters_correctly(self):
        db = TermPartitionedDB(4, ["color"], partition_by="range")
        db.put("p1", {"color": "apple"})
        db.put("p2", {"color": "banana"})
        db.put("p3", {"color": "cherry"})
        db.put("p4", {"color": "mango"})
        r = db.query_range("color", "a", "d")
        pks = {pk for pk, _ in r.data}
        assert {"p1", "p2", "p3"} == pks
        assert "p4" not in pks


# --- 6. compare_strategies ---
class TestCompareStrategies:
    def test_tradeoff_pattern(self):
        products = [
            ("p1", {"name": "Widget", "color": "red", "size": "large"}),
            ("p2", {"name": "Gadget", "color": "blue", "size": "small"}),
            ("p3", {"name": "Doohickey", "color": "red", "size": "small"}),
            ("p4", {"name": "Thingamajig", "color": "green", "size": "large"}),
            ("p5", {"name": "Whatsit", "color": "red", "size": "medium"}),
        ]
        queries = [("color", "red"), ("color", "blue"), ("size", "large")]
        result = compare_strategies(products, queries, 4, ["color", "size"])
        doc = result["document_partitioned"]
        term = result["term_partitioned"]
        # Doc-partitioned: writes cheap (1 partition), queries expensive (all N)
        assert doc["avg_partitions_per_write"] == 1.0
        assert doc["avg_partitions_per_query"] == 4.0
        # Term-partitioned: writes more expensive, queries cheaper
        assert term["avg_partitions_per_write"] > 1.0
        assert term["avg_partitions_per_query"] < 4.0
        assert "summary" in result


# --- 7. Scale + edge cases ---
class TestScaleAndEdgeCases:
    def test_1000_documents(self):
        colors = ["red", "blue", "green", "yellow", "black"]
        db = DocumentPartitionedDB(4, ["color"])
        for i in range(1000):
            db.put(f"d{i}", {"color": colors[i % 5]})
        for c in colors:
            assert len(db.query_by_field("color", c).data) == 200

    def test_missing_indexed_field(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"name": "no-color"})
        r = db.query_by_field("color", "red")
        assert len(r.data) == 0
        assert db.get("p1").data["name"] == "no-color"

    def test_hash_distribution(self):
        db = DocumentPartitionedDB(4, [])
        for i in range(100):
            db.put(f"doc{i}", {"val": i})
        counts = [db.get_partition(i).document_count for i in range(4)]
        assert sum(counts) == 100
        assert sum(1 for c in counts if c > 0) >= 2
