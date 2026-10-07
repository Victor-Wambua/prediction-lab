import time

from aviator_lab.collector import Collector
from aviator_lab.live import LivePredictor
from aviator_lab.report import quality_checks
from aviator_lab.simulation import generate_random_rounds


def test_live_predictions_scored_on_next_round_and_voided_on_gap(conn):
    x = list(generate_random_rounds(80, seed=3))
    c = Collector(conn, "test", LivePredictor())
    strip = list(reversed(x[:30]))  # newest first
    c.process_snapshot(strip)
    # Feed 10 more rounds one at a time, as the strip would show them.
    for i in range(30, 40):
        time.sleep(0.005)  # real rounds are seconds apart; avoid same-millisecond stamps
        strip = [x[i]] + strip[:-1]
        c.process_snapshot(strip)

    rows = conn.execute(
        "SELECT p.*, b.seq bseq, t.seq tseq, b.segment_id bseg, t.segment_id tseg FROM predictions p"
        " JOIN rounds b ON b.id=p.base_round_pk LEFT JOIN rounds t ON t.id=p.target_round_pk"
        " WHERE p.run_id=0").fetchall()
    resolved = [r for r in rows if r["status"] == "resolved"]
    assert len(resolved) == 10 * 4  # 10 rounds x 4 models below the logistic min_train
    for r in resolved:
        assert r["tseq"] == r["bseq"] + 1 and r["tseg"] == r["bseg"]
    assert quality_checks(conn)["predictions_after_target"] == 0

    # Gap: unrelated strip -> pending predictions are voided, not mis-scored.
    c.process_snapshot([9.99, 8.88, 7.77, 6.66, 5.55, 4.44, 3.33, 2.22, 1.11, 1.01])
    statuses = dict(conn.execute(
        "SELECT status, COUNT(*) FROM predictions WHERE run_id=0 GROUP BY status").fetchall())
    assert statuses["void"] == 4 and statuses["resolved"] == 40
