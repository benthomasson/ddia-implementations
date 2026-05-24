"""Tests for secondary index partitioning."""
import pytest
from secondary_index_partitioning import (
    Document, Partition, OperationResult,
    DocumentPartitionedDB, TermPartitionedDB, compare_strategies,
)


# 1. Basic CRUD
class TestBasicCRUD:
    def test_put_and_get(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"name": "Widget", "color": "red"})
        result = db.get("p1")
        assert result.data["name"] == "Widget"
        assert result.data["color"] == "red"
        assert result.partitions_touched == 1

    def test_get_nonexistent(self):
        db = DocumentPartitionedDB(4, ["color"])
        result = db.get("missing")
        assert result.data is None

    def test_delete(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"name": "Widget", "color": "red"})
        db.delete("p1")
        result = db.get("p1")
        assert result.data is None

    def test_update(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"name": "Widget", "color": "red"})
        db.put("p1", {"name": "Widget v2", "color": "blue"})
        result = db.get("p1")
        assert result.data["name"] == "Widget v2"
        assert result.data["color"] == "blue"

    def test_term_crud(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"name": "Widget", "color": "red"})
        result = db.get("p1")
        assert result.data["color"] == "red"
        db.delete("p1")
        assert db.get("p1").data is None


# 2. Hash partitioning distribution
class TestHashPartitioning:
    def test_documents_distribute(self):
        db = DocumentPartitionedDB(4, [])
        for i in range(100):
            db.put(f"doc{i}", {"val": i})
        counts = [db.get_partition(i).document_count for i in range(4)]
        assert sum(counts) == 100
        # At least 2 partitions should have documents (very likely with 100 docs)
        assert sum(1 for c in counts if c > 0) >= 2


# 3. Document-partitioned index correctness
class TestDocPartitionedIndex:
    def test_local_index_on_insert(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        # Find which partition has p1
        pid = hash("p1") % 4
        p = db.get_partition(pid)
        assert "p1" in p.local_index["color"]["red"]

    def test_local_index_on_update(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p1", {"color": "blue"})
        pid = hash("p1") % 4
        p = db.get_partition(pid)
        assert "p1" in p.local_index["color"]["blue"]
        assert "red" not in p.local_index["color"] or "p1" not in p.local_index["color"].get("red", set())

    def test_local_index_on_delete(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.delete("p1")
        pid = hash("p1") % 4
        p = db.get_partition(pid)
        assert "red" not in p.local_index["color"] or "p1" not in p.local_index["color"].get("red", set())


# 4. Document-partitioned query scatter/gather
class TestDocPartitionedQuery:
    def test_scatter_gather(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p2", {"color": "blue"})
        db.put("p3", {"color": "red"})
        result = db.query_by_field("color", "red")
        assert len(result.data) == 2
        assert result.partitions_touched == 4  # always all partitions

    def test_query_returns_correct_docs(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p2", {"color": "blue"})
        result = db.query_by_field("color", "red")
        pks = {pk for pk, _ in result.data}
        assert pks == {"p1"}


# 5. Term-partitioned index placement
class TestTermPartitionedIndex:
    def test_index_on_correct_partition(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        tid = hash("red") % 4
        p = db.get_partition(tid)
        assert "p1" in p.global_index["color"]["red"]


# 6. Term-partitioned query efficiency
class TestTermPartitionedQuery:
    def test_query_fewer_partitions(self):
        db = TermPartitionedDB(4, ["color"])
        for i in range(20):
            db.put(f"p{i}", {"color": "red"})
        result = db.query_by_field("color", "red")
        assert len(result.data) == 20
        assert result.partitions_touched <= 4


# 7. Write cost comparison
class TestWriteCost:
    def test_doc_write_touches_one(self):
        db = DocumentPartitionedDB(4, ["color", "size"])
        r = db.put("p1", {"color": "red", "size": "large"})
        assert r.partitions_touched == 1

    def test_term_write_touches_more(self):
        db = TermPartitionedDB(4, ["color", "size"])
        r = db.put("p1", {"color": "red", "size": "large"})
        assert r.partitions_touched >= 1


# 8. Document update index maintenance
class TestUpdateIndex:
    def test_old_entries_removed_on_update(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p1", {"color": "blue"})
        # Query old value should not find p1
        result = db.query_by_field("color", "red")
        pks = {pk for pk, _ in result.data}
        assert "p1" not in pks
        # Query new value should find p1
        result = db.query_by_field("color", "blue")
        pks = {pk for pk, _ in result.data}
        assert "p1" in pks

    def test_doc_db_update_index(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.put("p1", {"color": "blue"})
        result = db.query_by_field("color", "red")
        assert len(result.data) == 0
        result = db.query_by_field("color", "blue")
        assert len(result.data) == 1


# 9. Delete index cleanup
class TestDeleteCleanup:
    def test_term_delete_cleanup(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.delete("p1")
        result = db.query_by_field("color", "red")
        assert len(result.data) == 0

    def test_doc_delete_cleanup(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        db.delete("p1")
        result = db.query_by_field("color", "red")
        assert len(result.data) == 0


# 10. Async index mode
class TestAsyncIndex:
    def test_stale_before_flush(self):
        db = TermPartitionedDB(4, ["color"], async_index=True)
        db.put("p1", {"color": "red"})
        result = db.query_by_field("color", "red")
        assert len(result.data) == 0  # stale

    def test_correct_after_flush(self):
        db = TermPartitionedDB(4, ["color"], async_index=True)
        db.put("p1", {"color": "red"})
        count = db.flush_index()
        assert count > 0
        result = db.query_by_field("color", "red")
        assert len(result.data) == 1

    def test_async_update(self):
        db = TermPartitionedDB(4, ["color"], async_index=True)
        db.put("p1", {"color": "red"})
        db.flush_index()
        db.put("p1", {"color": "blue"})
        db.flush_index()
        assert len(db.query_by_field("color", "red").data) == 0
        assert len(db.query_by_field("color", "blue").data) == 1


# 11. Range query
class TestRangeQuery:
    def test_range_query(self):
        db = TermPartitionedDB(4, ["color"], partition_by="range")
        db.put("p1", {"color": "apple"})
        db.put("p2", {"color": "banana"})
        db.put("p3", {"color": "cherry"})
        db.put("p4", {"color": "mango"})
        db.put("p5", {"color": "zebra"})
        result = db.query_range("color", "a", "d")
        pks = {pk for pk, _ in result.data}
        assert "p1" in pks  # apple
        assert "p2" in pks  # banana
        assert "p3" in pks  # cherry
        assert "p4" not in pks  # mango
        assert "p5" not in pks  # zebra


# 12. Compare strategies
class TestCompareStrategies:
    def test_compare(self):
        products = [
            ("p1", {"name": "Widget", "color": "red", "size": "large"}),
            ("p2", {"name": "Gadget", "color": "blue", "size": "small"}),
            ("p3", {"name": "Doohickey", "color": "red", "size": "small"}),
            ("p4", {"name": "Thingamajig", "color": "green", "size": "large"}),
            ("p5", {"name": "Whatsit", "color": "red", "size": "medium"}),
        ]
        queries = [("color", "red"), ("color", "blue"), ("size", "large")]
        result = compare_strategies(products, queries, 4, ["color", "size"])
        doc_stats = result["document_partitioned"]
        term_stats = result["term_partitioned"]
        assert doc_stats["avg_partitions_per_write"] == 1.0
        assert doc_stats["avg_partitions_per_query"] == 4.0
        assert term_stats["avg_partitions_per_write"] > 1.0
        assert term_stats["avg_partitions_per_query"] < 4.0
        assert "summary" in result


# 13. Many documents
class TestManyDocuments:
    def test_1000_documents(self):
        colors = ["red", "blue", "green", "yellow", "black"]
        db_doc = DocumentPartitionedDB(4, ["color"])
        db_term = TermPartitionedDB(4, ["color"])
        for i in range(1000):
            fields = {"color": colors[i % len(colors)], "val": i}
            db_doc.put(f"d{i}", fields)
            db_term.put(f"d{i}", fields)
        # Each color should have 200 docs
        for color in colors:
            r1 = db_doc.query_by_field("color", color)
            r2 = db_term.query_by_field("color", color)
            assert len(r1.data) == 200
            assert len(r2.data) == 200


# 14. Edge cases
class TestEdgeCases:
    def test_query_nonexistent_value(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red"})
        result = db.query_by_field("color", "purple")
        assert len(result.data) == 0

    def test_missing_indexed_field(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"name": "Widget"})  # no color field
        result = db.query_by_field("color", "red")
        assert len(result.data) == 0
        # Document should still be stored
        assert db.get("p1").data["name"] == "Widget"

    def test_update_non_indexed_field(self):
        db = DocumentPartitionedDB(4, ["color"])
        db.put("p1", {"color": "red", "name": "Widget"})
        db.put("p1", {"color": "red", "name": "Widget v2"})
        result = db.query_by_field("color", "red")
        assert len(result.data) == 1
        assert result.data[0][1]["name"] == "Widget v2"

    def test_term_query_nonexistent(self):
        db = TermPartitionedDB(4, ["color"])
        result = db.query_by_field("color", "purple")
        assert len(result.data) == 0

    def test_term_missing_indexed_field(self):
        db = TermPartitionedDB(4, ["color"])
        db.put("p1", {"name": "Widget"})
        assert db.get("p1").data["name"] == "Widget"
        result = db.query_by_field("color", "red")
        assert len(result.data) == 0


# Test the example usage from the spec
class TestExampleUsage:
    def test_full_example(self):
        doc_db = DocumentPartitionedDB(4, ["color", "size"])
        term_db = TermPartitionedDB(4, ["color", "size"])

        products = [
            ("p1", {"name": "Widget", "color": "red", "size": "large"}),
            ("p2", {"name": "Gadget", "color": "blue", "size": "small"}),
            ("p3", {"name": "Doohickey", "color": "red", "size": "small"}),
            ("p4", {"name": "Thingamajig", "color": "green", "size": "large"}),
            ("p5", {"name": "Whatsit", "color": "red", "size": "medium"}),
        ]

        for pk, fields in products:
            r1 = doc_db.put(pk, fields)
            r2 = term_db.put(pk, fields)
            assert r1.partitions_touched == 1
            assert r2.partitions_touched >= 1

        result_doc = doc_db.query_by_field("color", "red")
        result_term = term_db.query_by_field("color", "red")

        assert len(result_doc.data) == 3
        assert len(result_term.data) == 3
        assert result_doc.partitions_touched == 4
        assert result_term.partitions_touched <= 4

    def test_async_example(self):
        async_db = TermPartitionedDB(4, ["color"], async_index=True)
        async_db.put("p1", {"name": "Widget", "color": "red"})
        result = async_db.query_by_field("color", "red")
        assert len(result.data) == 0
        async_db.flush_index()
        result = async_db.query_by_field("color", "red")
        assert len(result.data) == 1
