"""Point predictions: "the next round reaches at least T".

For each chance level L (50%, 30%, 20%, 10%), T is the highest multiplier that
at least a fraction L of all previous rounds reached (the empirical survival
quantile). A hit (green) is actual >= T, a miss (red) is actual < T.

If the past tells us nothing about the next round, each level's hit rate should
converge to L itself. A level that hits clearly more often than L over many
rounds would be evidence of a real signal; one that hits about L is not.
Predictions are logged before the round and scored after it.
"""
import math

import numpy as np

from . import database as db

LEVELS = (0.5, 0.3, 0.2, 0.1)
RTP = 0.97

SCHEMA = """
CREATE TABLE IF NOT EXISTS target_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    level            REAL NOT NULL,          -- intended hit chance, e.g. 0.5
    target           REAL NOT NULL,          -- predicted "at least" multiplier
    predicted_at     TEXT NOT NULL,
    base_round_pk    INTEGER NOT NULL REFERENCES rounds(id),
    target_round_pk  INTEGER REFERENCES rounds(id),
    status           TEXT NOT NULL DEFAULT 'pending',
    actual_multiplier REAL,
    hit              INTEGER,
    resolved_at      TEXT,
    UNIQUE (level, base_round_pk)
);
CREATE INDEX IF NOT EXISTS ix_target_predictions_status ON target_predictions(status);
"""


def ensure_schema(conn):
    conn.executescript(SCHEMA)


def empirical_target(x_past, level) -> float:
    """Highest t such that at least `level` of past rounds were >= t."""
    s = np.sort(np.asarray(x_past, float))[::-1]
    k = max(1, math.ceil(level * len(s)))
    return float(s[k - 1])


def fair_game_target(level, rtp=RTP) -> float:
    """Same idea under the assumed fair game: P(X >= t) = rtp / t."""
    return max(1.0, math.floor(round(rtp / level * 100, 6)) / 100)


def log_predictions(conn, base_round_pk, x_past):
    out = {}
    for level in LEVELS:
        t = empirical_target(x_past, level)
        conn.execute(
            "INSERT OR IGNORE INTO target_predictions (level, target, predicted_at, base_round_pk)"
            " VALUES (?,?,?,?)", (level, t, db.utcnow(), base_round_pk))
        out[level] = t
    return out


def resolve_pending(conn):
    pending = conn.execute(
        "SELECT p.id, p.target, r.segment_id AS seg, r.seq AS seq FROM target_predictions p"
        " JOIN rounds r ON r.id = p.base_round_pk WHERE p.status='pending'").fetchall()
    for p in pending:
        nxt = conn.execute("SELECT id, multiplier FROM rounds WHERE segment_id=? AND seq=?",
                           (p["seg"], p["seq"] + 1)).fetchone()
        if nxt is None:
            continue
        conn.execute(
            "UPDATE target_predictions SET status='resolved', target_round_pk=?, actual_multiplier=?,"
            " hit=?, resolved_at=? WHERE id=?",
            (nxt["id"], nxt["multiplier"], int(nxt["multiplier"] >= p["target"] - 1e-9),
             db.utcnow(), p["id"]))


def void_pending(conn):
    conn.execute("UPDATE target_predictions SET status='void', resolved_at=? WHERE status='pending'",
                 (db.utcnow(),))
