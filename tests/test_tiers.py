import time

import numpy as np

from aviator_lab import tiers
from aviator_lab.collector import Collector
from aviator_lab.live import LivePredictor
from aviator_lab.simulation import generate_random_rounds


def test_tier_boundaries():
    assert tiers.to_tier([1.00, 2.00, 2.01, 5.00, 5.01, 100]).tolist() == [0, 0, 1, 1, 2, 2]


def test_theoretical_tiers_match_simulation():
    x = generate_random_rounds(200_000, seed=0)
    freq = np.bincount(tiers.to_tier(x), minlength=3) / len(x)
    assert np.allclose(freq, tiers.theoretical_tier_probs(), atol=0.006)


def test_tier_predictions_logged_then_scored_on_next_round(conn):
    x = list(generate_random_rounds(50, seed=2))
    c = Collector(conn, "test", LivePredictor())
    strip = list(reversed(x[:30]))
    c.process_snapshot(strip)
    for v in x[30:35]:
        time.sleep(0.005)
        strip = [v] + strip[:-1]
        c.process_snapshot(strip)
    rows = conn.execute(
        "SELECT p.*, b.seq bseq, t.seq tseq, t.observed_at tobs FROM tier_predictions p"
        " JOIN rounds b ON b.id=p.base_round_pk LEFT JOIN rounds t ON t.id=p.target_round_pk").fetchall()
    resolved = [r for r in rows if r["status"] == "resolved"]
    assert len(resolved) == 5 * 4 and len(rows) == 6 * 4
    for r in resolved:
        assert r["tseq"] == r["bseq"] + 1
        assert r["predicted_at"] < r["tobs"]
        assert r["actual_tier"] == tiers.to_tier(r["actual_multiplier"])
        assert abs(r["p_low"] + r["p_mid"] + r["p_high"] - 1) < 1e-9
