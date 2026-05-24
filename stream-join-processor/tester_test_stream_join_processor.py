"""Tests for stream join processor."""

import pytest
from stream_join_processor import (
    StreamEvent, TimeWindow, StreamJoinProcessor, JoinType,
    JoinResult, JoinStats, TumblingWindowAggregator,
)


def ev(stream, key, timestamp, value=None):
    return StreamEvent(stream, key, value or {}, timestamp)


# --- Spec example: click-through rate with LEFT join ---

class TestSpecExamples:
    def test_impressions_clicks_left_join(self):
        """Spec example: join impressions with clicks, LEFT join."""
        window = TimeWindow(duration_seconds=10.0)
        processor = StreamJoinProcessor(
            left_stream="impressions", right_stream="clicks",
            window=window, join_type=JoinType.LEFT,
        )

        # Impression at t=1
        processor.process_event(StreamEvent(
            stream_name="impressions", key="ad:100",
            value={"ad_id": 100, "page": "/home"}, timestamp=1.0,
        ))

        # Click at t=3 (within 10s window) -> match
        results = processor.process_event(StreamEvent(
            stream_name="clicks", key="ad:100",
            value={"ad_id": 100, "click_pos": "top"}, timestamp=3.0,
        ))
        assert len(results) == 1
        assert results[0].left_event.value["page"] == "/home"
        assert results[0].right_event.value["click_pos"] == "top"

        # Another impression with no click
        processor.process_event(StreamEvent(
            stream_name="impressions", key="ad:200",
            value={"ad_id": 200, "page": "/about"}, timestamp=5.0,
        ))

        # Advance past window -> miss for ad:200
        misses = processor.advance_time(20.0)
        assert any(r.key == "ad:200" and r.right_event is None for r in misses)

        stats = processor.stats
        assert stats.left_events_processed == 2
        assert stats.right_events_processed == 1
        assert stats.matches_emitted == 1
        assert stats.misses_emitted >= 1

    def test_inner_join_orders_payments(self):
        """Spec example: inner join orders with payments."""
        p = StreamJoinProcessor(
            left_stream="orders", right_stream="payments",
            window=TimeWindow(5.0), join_type=JoinType.INNER,
        )
        p.process_event(StreamEvent("orders", "order:1",
            {"item": "book", "price": 15}, timestamp=100.0))
        results = p.process_event(StreamEvent("payments", "order:1",
            {"method": "credit", "amount": 15}, timestamp=103.0))
        assert len(results) == 1

        # order:2 with no payment within window
        p.process_event(StreamEvent("orders", "order:2",
            {"item": "pen", "price": 5}, timestamp=110.0))
        p.advance_time(120.0)
        assert p.stats.matches_emitted == 1  # still just 1

    def test_late_event_handling(self):
        """Spec example: late event dropped."""
        p = StreamJoinProcessor(
            left_stream="a", right_stream="b",
            window=TimeWindow(5.0), join_type=JoinType.INNER,
            allowed_lateness=2.0,
        )
        p.process_event(StreamEvent("a", "k1", {}, timestamp=100.0))
        p.advance_time(110.0)
        # Event at t=107 is late (watermark=110, cutoff=108)
        p.process_event(StreamEvent("b", "k1", {}, timestamp=107.0))
        assert p.stats.late_events_dropped == 1


# --- Core join semantics ---

class TestInnerJoin:
    def test_match_within_window(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        p.process_event(ev("L", "k1", 100.0, {"a": 1}))
        results = p.process_event(ev("R", "k1", 103.0, {"b": 2}))
        assert len(results) == 1
        assert results[0].key == "k1"
        assert results[0].left_event.value == {"a": 1}
        assert results[0].right_event.value == {"b": 2}

    def test_no_match_different_key(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        p.process_event(ev("L", "k1", 100.0))
        results = p.process_event(ev("R", "k2", 103.0))
        assert len(results) == 0

    def test_no_match_outside_window(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        p.process_event(ev("L", "k1", 100.0))
        results = p.process_event(ev("R", "k1", 106.0))
        assert len(results) == 0

    def test_multiple_right_matches(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        p.process_event(ev("R", "k1", 100.0, {"r": 1}))
        p.process_event(ev("R", "k1", 102.0, {"r": 2}))
        # L event at t=103 (after watermark=102, so not late) matches both R events
        results = p.process_event(ev("L", "k1", 103.0, {"l": 1}))
        assert len(results) == 2


class TestLeftJoin:
    def test_miss_on_expiration(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.LEFT)
        p.process_event(ev("L", "k1", 100.0, {"x": 1}))
        misses = p.advance_time(110.0)
        assert len(misses) == 1
        assert misses[0].left_event is not None
        assert misses[0].right_event is None

    def test_matched_no_miss(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.LEFT)
        p.process_event(ev("L", "k1", 100.0))
        results = p.process_event(ev("R", "k1", 103.0))
        assert len(results) == 1
        # No miss on expiration since matched
        misses = p.advance_time(110.0)
        assert not any(r.key == "k1" and r.right_event is None for r in misses)


class TestFullOuterJoin:
    def test_misses_both_sides(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.FULL_OUTER)
        p.process_event(ev("L", "k1", 100.0))
        p.process_event(ev("R", "k2", 101.0))
        misses = p.advance_time(110.0)
        left_miss = [r for r in misses if r.key == "k1"]
        right_miss = [r for r in misses if r.key == "k2"]
        assert len(left_miss) == 1 and left_miss[0].right_event is None
        assert len(right_miss) == 1 and right_miss[0].left_event is None


# --- Window, lateness, buffers ---

class TestWindowAndBuffers:
    def test_expiration_clears_buffer(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        p.process_event(ev("L", "k1", 100.0))
        assert p.buffer_size == (1, 0)
        p.advance_time(106.0)
        assert p.buffer_size == (0, 0)

    def test_out_of_order_within_lateness(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER, allowed_lateness=3.0)
        p.process_event(ev("L", "k1", 100.0))
        p.process_event(ev("R", "k1", 105.0))  # watermark=105
        # Late event at 103, cutoff=105-3=102, 103>=102 so accepted
        results = p.process_event(ev("L", "k1", 103.0))
        assert p.stats.late_events_dropped == 0
        assert len(results) == 1  # matches R:k1@105

    def test_buffer_bounded_after_many_events(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        for i in range(1000):
            t = float(i)
            p.process_event(ev("L", f"k{i}", t))
            p.process_event(ev("R", f"k{i}", t + 1.0))
        left, right = p.buffer_size
        assert left <= 10
        assert right <= 10


# --- Callback and aggregation ---

class TestCallbackAndAggregation:
    def test_callback_invoked(self):
        received = []
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER,
                                on_result=lambda r: received.append(r))
        p.process_event(ev("L", "k1", 100.0))
        p.process_event(ev("R", "k1", 102.0))
        assert len(received) == 1
        assert received[0].key == "k1"

    def test_tumbling_window_aggregation(self):
        agg = TumblingWindowAggregator(10.0, lambda key, results: len(results))
        agg.add(JoinResult("k1", None, None, 3.0))
        agg.add(JoinResult("k1", None, None, 7.0))
        agg.add(JoinResult("k1", None, None, 13.0))
        closed = agg.advance_time(20.0)
        w0 = [c for c in closed if c[1] == 0.0 and c[0] == "k1"]
        w1 = [c for c in closed if c[1] == 10.0 and c[0] == "k1"]
        assert len(w0) == 1 and w0[0][3] == 2
        assert len(w1) == 1 and w1[0][3] == 1


# --- Statistics ---

class TestStatistics:
    def test_all_counters(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(10.0), JoinType.LEFT)
        p.process_event(ev("L", "k1", 1.0))
        p.process_event(ev("R", "k1", 3.0))
        p.process_event(ev("L", "k2", 5.0))  # no right match
        p.advance_time(20.0)

        s = p.stats
        assert s.left_events_processed == 2
        assert s.right_events_processed == 1
        assert s.matches_emitted == 1
        assert s.misses_emitted >= 1

    def test_get_results_clears(self):
        p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
        p.process_event(ev("L", "k1", 100.0))
        p.process_event(ev("R", "k1", 102.0))
        r1 = p.get_results()
        assert len(r1) == 1
        r2 = p.get_results()
        assert len(r2) == 0
