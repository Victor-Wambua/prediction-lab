import sqlite3

import numpy as np
import pytest

from prediction_lab import database as db


def test_append_and_order(conn):
    seg = db.new_segment(conn, "t", "first run")
    db.append_rounds(conn, seg, [1.0, 2.0, 3.0], "t")
    db.append_rounds(conn, seg, [4.0], "t")
    df = db.load_rounds(conn)
    assert df.multiplier.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert df.seq.tolist() == [0, 1, 2, 3]
    assert [r["multiplier"] for r in db.segment_tail(conn, seg, 2)] == [4.0, 3.0]


def test_constraints(conn):
    seg = db.new_segment(conn, "t", "x")
    with pytest.raises(sqlite3.IntegrityError):
        db.append_rounds(conn, seg, [0.5], "t")  # multiplier < 1
    conn.execute("INSERT INTO rounds (segment_id, seq, round_id, multiplier, observed_at, source)"
                 " VALUES (?,?,?,?,?,?)", (seg, 10, "R1", 1.5, db.utcnow(), "t"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO rounds (segment_id, seq, round_id, multiplier, observed_at, source)"
                     " VALUES (?,?,?,?,?,?)", (seg, 11, "R1", 1.5, db.utcnow(), "t"))


def test_prediction_insert_is_idempotent_and_resolves(conn):
    seg = db.new_segment(conn, "t", "x")
    (pk,) = db.append_rounds(conn, seg, [1.2], "t")
    p = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    db.insert_prediction(conn, "m", pk, p)
    db.insert_prediction(conn, "m", pk, p)  # duplicate ignored
    rows = conn.execute("SELECT * FROM predictions").fetchall()
    assert len(rows) == 1 and rows[0]["predicted_class"] == 0
    (pk2,) = db.append_rounds(conn, seg, [1.3], "t")
    db.resolve_prediction(conn, rows[0]["id"], pk2, 1.3, 0, p)
    r = conn.execute("SELECT * FROM predictions").fetchone()
    assert r["status"] == "resolved" and r["correct"] == 1
    assert r["log_loss"] == pytest.approx(-np.log(0.4))
