from aviator_lab import database as db
from aviator_lab.collector import Collector
from aviator_lab.history import align, parse_multiplier, parse_snapshot


def test_parse_multiplier_strict():
    assert parse_multiplier("1.37x") == 1.37
    assert parse_multiplier(" 15.19X ") == 15.19
    assert parse_multiplier("1,234.50x") == 1234.5
    assert parse_multiplier("2x") == 2.0
    for bad in ["", "abc", "0.99x", "1.375x", "Cash out 2.00x", "1.37", None]:
        assert parse_multiplier(bad) is None


def test_snapshot_rejected_if_any_item_bad():
    assert parse_snapshot(["1.37x", "2.00x"]) == [1.37, 2.0]
    assert parse_snapshot(["1.37x", "oops"]) is None
    assert parse_snapshot([]) is None


TAIL = [1.37, 2.10, 1.00, 1.37, 5.20, 1.00, 1.00, 3.30, 1.37, 1.12]  # newest first, repeats


def test_align_detects_new_rounds_even_when_values_repeat():
    # Two new rounds whose values ALSO appear in the tail (V1 would drop them).
    snap = [1.00, 1.37] + TAIL[:8]
    a = align(snap, TAIL, min_overlap=8)
    assert a.new_count == 2


def test_align_no_change():
    assert align(TAIL, TAIL, 8).new_count == 0


def test_align_gap_when_no_overlap():
    snap = [9.99, 8.88, 7.77, 6.66, 5.55, 4.44, 3.33, 2.22, 1.11, 1.01]
    assert align(snap, TAIL, 8).new_count is None


def test_align_gap_when_overlap_too_short():
    snap = [9.0] * 5 + TAIL[:5]  # only 5 overlapping < 8 required
    assert align(snap, TAIL, 8).new_count is None


def test_align_flags_ambiguity_on_constant_runs():
    tail = [1.00] * 12
    a = align([1.00] * 12, tail, 8)
    assert a.new_count == 0 and a.ambiguous


def _vals(conn, seg):
    return [r["multiplier"] for r in conn.execute(
        "SELECT multiplier FROM rounds WHERE segment_id=? ORDER BY seq", (seg,))]


def test_collector_bootstrap_append_refresh_restart(conn):
    c = Collector(conn, "test")
    c.process_snapshot(TAIL)
    seg = c.segment
    assert _vals(conn, seg) == list(reversed(TAIL))

    # Same snapshot again (page refresh): nothing new.
    c.last_processed = None
    assert c.process_snapshot(TAIL) == []

    # New round with a repeated value.
    snap2 = [1.37] + TAIL[:-1]
    assert len(c.process_snapshot(snap2)) == 1

    # Collector restart: fresh object, state comes from the DB.
    c2 = Collector(conn, "test")
    snap3 = [1.00, 1.00] + snap2[:-2]
    assert len(c2.process_snapshot(snap3)) == 2
    assert c2.segment == seg
    assert _vals(conn, seg) == list(reversed(TAIL)) + [1.37, 1.00, 1.00]


def test_collector_gap_opens_new_segment(conn):
    c = Collector(conn, "test")
    c.process_snapshot(TAIL)
    first = c.segment
    other = [9.99, 8.88, 7.77, 6.66, 5.55, 4.44, 3.33, 2.22, 1.11, 1.01]
    c.process_snapshot(other)
    assert c.segment != first
    reason = conn.execute("SELECT reason FROM segments WHERE id=?", (c.segment,)).fetchone()[0]
    assert reason.startswith("gap")
