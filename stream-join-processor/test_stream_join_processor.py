"""Tests for stream join processor."""

import pytest
from stream_join_processor import (
    StreamEvent, TimeWindow, StreamJoinProcessor, JoinType,
    JoinResult, TumblingWindowAggregator,
)


def make_event(stream, key, timestamp, value=None):
    return StreamEvent(stream, key, value or {}, timestamp)


# 1. Inner join: matching events within window produce a result
def test_inner_join_match():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
    p.process_event(make_event("L", "k1", 100.0, {"a": 1}))
    results = p.process_event(make_event("R", "k1", 103.0, {"b": 2}))
    assert len(results) == 1
    assert results[0].key == "k1"
    assert results[0].left_event.value == {"a": 1}
    assert results[0].right_event.value == {"b": 2}


# 2. Inner join: no match (different key or outside window)
def test_inner_join_no_match():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
    p.process_event(make_event("L", "k1", 100.0))
    # Different key
    results = p.process_event(make_event("R", "k2", 103.0))
    assert len(results) == 0
    # Same key but outside window
    p.process_event(make_event("L", "k3", 100.0))
    results = p.process_event(make_event("R", "k3", 106.0))
    assert len(results) == 0


# 3. Left join: unmatched left events emit miss on expiration
def test_left_join_miss():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.LEFT)
    p.process_event(make_event("L", "k1", 100.0, {"x": 1}))
    misses = p.advance_time(110.0)
    assert len(misses) == 1
    assert misses[0].key == "k1"
    assert misses[0].left_event is not None
    assert misses[0].right_event is None


# 4. Left join: matched left events emit normal result
def test_left_join_match():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.LEFT)
    p.process_event(make_event("L", "k1", 100.0))
    results = p.process_event(make_event("R", "k1", 103.0))
    assert len(results) == 1
    assert results[0].left_event is not None
    assert results[0].right_event is not None
    # Should NOT emit miss on expiration since it was matched
    misses = p.advance_time(110.0)
    assert not any(r.key == "k1" and r.right_event is None for r in misses)


# 5. Full outer join: unmatched events on either side emit misses
def test_full_outer_join_misses():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.FULL_OUTER)
    p.process_event(make_event("L", "k1", 100.0))
    p.process_event(make_event("R", "k2", 101.0))
    misses = p.advance_time(110.0)
    left_miss = [r for r in misses if r.key == "k1"]
    right_miss = [r for r in misses if r.key == "k2"]
    assert len(left_miss) == 1 and left_miss[0].right_event is None
    assert len(right_miss) == 1 and right_miss[0].left_event is None


# 6. Window expiration: old events removed from buffers
def test_window_expiration():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
    p.process_event(make_event("L", "k1", 100.0))
    assert p.buffer_size == (1, 0)
    p.advance_time(106.0)
    assert p.buffer_size == (0, 0)


# 7. Out-of-order events within allowed_lateness are processed
def test_out_of_order_within_lateness():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER, allowed_lateness=3.0)
    p.process_event(make_event("L", "k1", 100.0))
    p.process_event(make_event("R", "k1", 105.0))  # watermark = 105
    # Late event at 103, within lateness (cutoff = 105 - 3 = 102)
    results = p.process_event(make_event("L", "k1", 103.0))
    assert p.stats.late_events_dropped == 0
    # Should match against R:k1@105 (|103-105| = 2 <= 5)
    assert len(results) == 1


# 8. Late events beyond allowed_lateness are dropped
def test_late_event_dropped():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER, allowed_lateness=2.0)
    p.process_event(make_event("L", "k1", 100.0))
    p.advance_time(110.0)
    # Event at 107, cutoff = 110 - 2 = 108, so 107 < 108 => dropped
    p.process_event(make_event("R", "k1", 107.0))
    assert p.stats.late_events_dropped == 1


# 9. Multiple matches: one left event matching multiple right events
def test_multiple_matches():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
    p.process_event(make_event("R", "k1", 100.0, {"r": 1}))
    p.process_event(make_event("R", "k1", 102.0, {"r": 2}))
    results = p.process_event(make_event("L", "k1", 101.0, {"l": 1}))
    assert len(results) == 2


# 10. Statistics accuracy
def test_statistics():
    p = StreamJoinProcessor("L", "R", TimeWindow(10.0), JoinType.LEFT)
    p.process_event(make_event("L", "k1", 1.0))
    p.process_event(make_event("R", "k1", 3.0))
    p.process_event(make_event("L", "k2", 5.0))
    p.advance_time(20.0)

    s = p.stats
    assert s.left_events_processed == 2
    assert s.right_events_processed == 1
    assert s.matches_emitted == 1
    assert s.misses_emitted >= 1  # k2 had no match


# 11. Callback mode
def test_callback_mode():
    received = []
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER,
                            on_result=lambda r: received.append(r))
    p.process_event(make_event("L", "k1", 100.0))
    p.process_event(make_event("R", "k1", 102.0))
    assert len(received) == 1
    assert received[0].key == "k1"


# 12. Tumbling window aggregation
def test_tumbling_window_aggregation():
    agg = TumblingWindowAggregator(10.0, lambda key, results: len(results))
    agg.add(JoinResult("k1", None, None, 3.0))
    agg.add(JoinResult("k1", None, None, 7.0))
    agg.add(JoinResult("k1", None, None, 13.0))

    closed = agg.advance_time(20.0)
    # Window [0, 10) should have 2 results, [10, 20) should have 1
    w0 = [c for c in closed if c[1] == 0.0 and c[0] == "k1"]
    w1 = [c for c in closed if c[1] == 10.0 and c[0] == "k1"]
    assert len(w0) == 1 and w0[0][3] == 2
    assert len(w1) == 1 and w1[0][3] == 1


# 13. Buffer cleanup: buffer sizes are bounded after many events
def test_buffer_cleanup():
    p = StreamJoinProcessor("L", "R", TimeWindow(5.0), JoinType.INNER)
    for i in range(1000):
        t = float(i)
        p.process_event(make_event("L", f"k{i}", t))
        p.process_event(make_event("R", f"k{i}", t + 1.0))

    # Buffer should only contain events within window of watermark (999+1=1000)
    left, right = p.buffer_size
    assert left <= 10  # window is 5s, so at most ~5-6 events per stream
    assert right <= 10
    p.get_results()  # clear results buffer


# Integration test from the spec examples
def test_spec_example():
    window = TimeWindow(duration_seconds=10.0)
    processor = StreamJoinProcessor(
        left_stream="impressions", right_stream="clicks",
        window=window, join_type=JoinType.LEFT,
    )

    processor.process_event(StreamEvent(
        stream_name="impressions", key="ad:100",
        value={"ad_id": 100, "page": "/home"}, timestamp=1.0,
    ))

    results = processor.process_event(StreamEvent(
        stream_name="clicks", key="ad:100",
        value={"ad_id": 100, "click_pos": "top"}, timestamp=3.0,
    ))

    assert len(results) == 1
    assert results[0].left_event.value["page"] == "/home"
    assert results[0].right_event.value["click_pos"] == "top"

    processor.process_event(StreamEvent(
        stream_name="impressions", key="ad:200",
        value={"ad_id": 200, "page": "/about"}, timestamp=5.0,
    ))

    misses = processor.advance_time(20.0)
    assert any(r.key == "ad:200" and r.right_event is None for r in misses)

    stats = processor.stats
    assert stats.left_events_processed == 2
    assert stats.right_events_processed == 1
    assert stats.matches_emitted == 1
    assert stats.misses_emitted >= 1
