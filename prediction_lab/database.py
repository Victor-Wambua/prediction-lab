"""SQLite storage.

Key ideas:
- `segments`: a contiguous run of rounds with no known gaps. If the collector
  cannot prove that a new history snapshot continues the stored tail (page was
  closed for a while, DOM glitch, ...) it starts a new segment. Sequence features
  never cross segment boundaries, and a live prediction is only scored against
  the round that immediately follows it in the same segment.
- `rounds.seq`: position within the segment (0-based, oldest first).
- `predictions`: every prediction, live (run_id=0) or backtest (run_id>0).
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .classes import N_CLASSES

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    source        TEXT NOT NULL,
    reason        TEXT NOT NULL            -- why a new segment was opened
);

CREATE TABLE IF NOT EXISTS rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id      INTEGER NOT NULL REFERENCES segments(id),
    seq             INTEGER NOT NULL,      -- order within segment, oldest = 0
    round_id        TEXT,                  -- site round id, if ever exposed
    multiplier      REAL NOT NULL CHECK (multiplier >= 1.0),
    observed_at     TEXT NOT NULL,         -- when the collector first saw it (UTC)
    round_timestamp TEXT,                  -- when the round happened, if exposed
    is_bootstrap    INTEGER NOT NULL DEFAULT 0, -- came from the initial history strip
    source          TEXT NOT NULL,
    raw_data        TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (segment_id, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_rounds_round_id ON rounds(round_id) WHERE round_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_rounds_observed_at ON rounds(observed_at);
CREATE INDEX IF NOT EXISTS ix_rounds_multiplier ON rounds(multiplier);

CREATE TABLE IF NOT EXISTS model_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    dataset       TEXT NOT NULL,           -- 'real' | 'synthetic-null' | 'synthetic-signal'
    code_version  TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    n_rounds      INTEGER NOT NULL,
    n_scored      INTEGER NOT NULL,
    results_json  TEXT NOT NULL            -- leaderboard + significance tests
);

CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL DEFAULT 0,  -- 0 = live; else model_runs.id
    model_name       TEXT NOT NULL,
    predicted_at     TEXT NOT NULL,
    base_round_pk    INTEGER NOT NULL REFERENCES rounds(id),  -- last round known
    target_round_pk  INTEGER REFERENCES rounds(id),           -- round being predicted
    p0 REAL NOT NULL, p1 REAL NOT NULL, p2 REAL NOT NULL, p3 REAL NOT NULL, p4 REAL NOT NULL,
    predicted_class  INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',  -- pending | resolved | void
    actual_multiplier REAL,
    actual_class     INTEGER,
    correct          INTEGER,
    log_loss         REAL,
    brier            REAL,
    resolved_at      TEXT,
    UNIQUE (run_id, model_name, base_round_pk)
);
CREATE INDEX IF NOT EXISTS ix_predictions_status ON predictions(status, run_id);
"""

PROB_COLS = [f"p{i}" for i in range(N_CLASSES)]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # dashboard can read while collector writes
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    from . import targets, tiers
    tiers.ensure_schema(conn)
    targets.ensure_schema(conn)
    return conn


# ---------------------------------------------------------------- segments/rounds

def new_segment(conn, source: str, reason: str) -> int:
    cur = conn.execute(
        "INSERT INTO segments (started_at, source, reason) VALUES (?,?,?)",
        (utcnow(), source, reason),
    )
    conn.commit()
    return cur.lastrowid


def latest_segment(conn) -> int | None:
    row = conn.execute("SELECT MAX(id) AS id FROM segments").fetchone()
    return row["id"]


def segment_tail(conn, segment_id: int, n: int) -> list[sqlite3.Row]:
    """Last n rounds of a segment, NEWEST FIRST (same order as the history strip)."""
    return conn.execute(
        "SELECT * FROM rounds WHERE segment_id=? ORDER BY seq DESC LIMIT ?",
        (segment_id, n),
    ).fetchall()


def append_rounds(conn, segment_id: int, multipliers_oldest_first, source: str,
                  is_bootstrap: bool = False, raw: dict | None = None) -> list[int]:
    """Append rounds to the end of a segment in one transaction. Returns new pks."""
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) AS s FROM rounds WHERE segment_id=?", (segment_id,)
    ).fetchone()
    seq = row["s"] + 1
    now = utcnow()
    pks = []
    with conn:
        for x in multipliers_oldest_first:
            cur = conn.execute(
                "INSERT INTO rounds (segment_id, seq, multiplier, observed_at, is_bootstrap, source, raw_data)"
                " VALUES (?,?,?,?,?,?,?)",
                (segment_id, seq, float(x), now, int(is_bootstrap), source,
                 json.dumps(raw) if raw else None),
            )
            pks.append(cur.lastrowid)
            seq += 1
    return pks


def load_rounds(conn):
    """All rounds as a DataFrame ordered by (segment, seq)."""
    import pandas as pd
    return pd.read_sql_query(
        "SELECT id, segment_id, seq, multiplier, observed_at, is_bootstrap FROM rounds"
        " ORDER BY segment_id, seq",
        conn,
    )


# ---------------------------------------------------------------- predictions

def insert_prediction(conn, model_name: str, base_round_pk: int, probs: np.ndarray,
                      run_id: int = 0, predicted_at: str | None = None,
                      commit: bool = True) -> None:
    probs = np.asarray(probs, dtype=float)
    conn.execute(
        f"INSERT OR IGNORE INTO predictions (run_id, model_name, predicted_at, base_round_pk,"
        f" {', '.join(PROB_COLS)}, predicted_class) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (run_id, model_name, predicted_at or utcnow(), base_round_pk,
         *map(float, probs), int(np.argmax(probs))),
    )
    if commit:
        conn.commit()


def resolve_prediction(conn, pred_id: int, target_round_pk: int, multiplier: float,
                       actual_class: int, probs: np.ndarray, commit: bool = True) -> None:
    from .metrics import per_row_brier, per_row_log_loss
    probs = np.asarray(probs, dtype=float)[None, :]
    y = np.array([actual_class])
    conn.execute(
        "UPDATE predictions SET status='resolved', target_round_pk=?, actual_multiplier=?,"
        " actual_class=?, correct=?, log_loss=?, brier=?, resolved_at=? WHERE id=?",
        (target_round_pk, multiplier, actual_class, int(np.argmax(probs[0]) == actual_class),
         float(per_row_log_loss(y, probs)[0]), float(per_row_brier(y, probs)[0]), utcnow(), pred_id),
    )
    if commit:
        conn.commit()


def void_pending(conn, run_id: int = 0) -> int:
    cur = conn.execute(
        "UPDATE predictions SET status='void', resolved_at=? WHERE status='pending' AND run_id=?",
        (utcnow(), run_id),
    )
    if run_id == 0:
        from . import targets, tiers
        tiers.void_pending(conn)
        targets.void_pending(conn)
    conn.commit()
    return cur.rowcount


def save_model_run(conn, dataset: str, cfg: dict, n_rounds: int, n_scored: int,
                   results: dict) -> int:
    from . import __version__
    cur = conn.execute(
        "INSERT INTO model_runs (created_at, dataset, code_version, config_json, n_rounds,"
        " n_scored, results_json) VALUES (?,?,?,?,?,?,?)",
        (utcnow(), dataset, __version__, json.dumps(cfg), n_rounds, n_scored,
         json.dumps(results, default=float)),
    )
    conn.commit()
    return cur.lastrowid
