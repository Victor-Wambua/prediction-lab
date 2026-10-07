"""Three-tier view of the next round, alongside the 5-class research pipeline.

    Low    : x <= 2.00
    Middle : 2.00 < x <= 5.00
    High   : x > 5.00

Multipliers are whole cents, so the tiers are computed on cents to avoid float
edge cases. Tier forecasts are logged before the round and scored after it,
exactly like the 5-class predictions, in their own table.
"""
import numpy as np

from . import database as db
from .metrics import EPS

TIER_LABELS = ["Low (≤2.00x)", "Middle (2.01–5.00x)", "High (>5.00x)"]
TIER_SHORT = ["Low", "Middle", "High"]
N_TIERS = 3
PRIOR_STRENGTH = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS tier_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name       TEXT NOT NULL,
    predicted_at     TEXT NOT NULL,
    base_round_pk    INTEGER NOT NULL REFERENCES rounds(id),
    target_round_pk  INTEGER REFERENCES rounds(id),
    p_low REAL NOT NULL, p_mid REAL NOT NULL, p_high REAL NOT NULL,
    predicted_tier   INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    actual_multiplier REAL,
    actual_tier      INTEGER,
    correct          INTEGER,
    log_loss         REAL,
    brier            REAL,
    resolved_at      TEXT,
    UNIQUE (model_name, base_round_pk)
);
CREATE INDEX IF NOT EXISTS ix_tier_predictions_status ON tier_predictions(status);
"""


def ensure_schema(conn):
    conn.executescript(SCHEMA)


def to_tier(x):
    cents = np.rint(np.asarray(x, dtype=float) * 100).astype(int)
    return np.where(cents <= 200, 0, np.where(cents <= 500, 1, 2))


def theoretical_tier_probs(rtp=0.97):
    """Under P(X >= m) = rtp/m on a cent grid: P(X > 2.00) = rtp/2.01, etc."""
    p_above_2 = rtp / 2.01
    p_above_5 = rtp / 5.01
    return np.array([1 - p_above_2, p_above_2 - p_above_5, p_above_5])


def _freq(t, prior=None, strength=0.0):
    c = np.bincount(np.asarray(t, dtype=int), minlength=N_TIERS).astype(float)
    if prior is None:
        c += 1.0
    else:
        c += strength * prior
    return c / c.sum()


def predict_all(x_all, x_segment):
    """Tier probabilities for the round after `x_segment` from each tier model."""
    t_all = to_tier(x_all)
    t_seg = to_tier(x_segment)
    glob = _freq(t_all)
    out = {
        "global_freq": glob,
        "recent_50": _freq(t_all[-50:], glob, PRIOR_STRENGTH),
        "theoretical_rtp97": theoretical_tier_probs(0.97),
    }
    # Markov: P(tier | previous tier), transitions counted within segments only.
    if len(t_seg):
        prev = t_seg[-1]
        T = np.zeros((N_TIERS, N_TIERS))
        np.add.at(T, (t_seg[:-1], t_seg[1:]), 1.0)
        row = T[prev] + PRIOR_STRENGTH * glob
        out["markov_1"] = row / row.sum()
    return out


def log_predictions(conn, base_round_pk, preds):
    for name, p in preds.items():
        conn.execute(
            "INSERT OR IGNORE INTO tier_predictions (model_name, predicted_at, base_round_pk,"
            " p_low, p_mid, p_high, predicted_tier) VALUES (?,?,?,?,?,?,?)",
            (name, db.utcnow(), base_round_pk, *map(float, p), int(np.argmax(p))),
        )


def resolve_pending(conn):
    pending = conn.execute(
        "SELECT p.*, r.segment_id AS seg, r.seq AS seq FROM tier_predictions p"
        " JOIN rounds r ON r.id = p.base_round_pk WHERE p.status='pending'").fetchall()
    for p in pending:
        target = conn.execute("SELECT * FROM rounds WHERE segment_id=? AND seq=?",
                              (p["seg"], p["seq"] + 1)).fetchone()
        if target is None:
            continue
        probs = np.array([p["p_low"], p["p_mid"], p["p_high"]])
        actual = int(to_tier(target["multiplier"]))
        ll = float(-np.log(max(probs[actual], EPS)))
        brier = float(((probs - np.eye(N_TIERS)[actual]) ** 2).sum())
        conn.execute(
            "UPDATE tier_predictions SET status='resolved', target_round_pk=?, actual_multiplier=?,"
            " actual_tier=?, correct=?, log_loss=?, brier=?, resolved_at=? WHERE id=?",
            (target["id"], target["multiplier"], actual, int(np.argmax(probs) == actual),
             ll, brier, db.utcnow(), p["id"]))


def void_pending(conn):
    conn.execute("UPDATE tier_predictions SET status='void', resolved_at=? WHERE status='pending'",
                 (db.utcnow(),))
