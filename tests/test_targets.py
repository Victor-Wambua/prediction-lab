import time

import numpy as np

from prediction_lab import targets
from prediction_lab.collector import Collector
from prediction_lab.live import LivePredictor
from prediction_lab.simulation import generate_random_rounds


def test_empirical_target_is_survival_quantile():
    x = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0]
    assert targets.empirical_target(x, 0.5) == 8.0   # 5 of 10 rounds reached >= 8.0
    assert targets.empirical_target(x, 0.1) == 100.0
    assert targets.empirical_target(x, 0.2) == 50.0
    assert targets.fair_game_target(0.5) == 1.94
    assert targets.fair_game_target(0.2) == 4.85


def test_hit_rates_match_levels_on_fair_data():
    """No signal in i.i.d. data: each level's hit rate should be about the level."""
    x = generate_random_rounds(6000, seed=9)
    for level in targets.LEVELS:
        hits = [x[t] >= targets.empirical_target(x[:t], level) for t in range(500, len(x))]
        assert abs(np.mean(hits) - level) < 0.03, (level, np.mean(hits))


def test_targets_logged_before_and_scored_on_next_round(conn):
    x = list(generate_random_rounds(45, seed=4))
    c = Collector(conn, "test", LivePredictor())
    strip = list(reversed(x[:30]))
    c.process_snapshot(strip)
    for v in x[30:]:
        time.sleep(0.005)
        strip = [v] + strip[:-1]
        c.process_snapshot(strip)
    rows = conn.execute(
        "SELECT p.*, b.seq bseq, t.seq tseq, t.observed_at tobs FROM target_predictions p"
        " JOIN rounds b ON b.id=p.base_round_pk LEFT JOIN rounds t ON t.id=p.target_round_pk"
        " WHERE p.status='resolved'").fetchall()
    assert len(rows) == 15 * len(targets.LEVELS)
    for r in rows:
        assert r["tseq"] == r["bseq"] + 1 and r["predicted_at"] < r["tobs"]
        assert r["hit"] == int(r["actual_multiplier"] >= r["target"])
