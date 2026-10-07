"""Data-quality and process report for the collected rounds."""
import numpy as np

from . import database as db, stats
from .classes import LABELS


def quality_checks(conn) -> dict:
    q = {}
    q["rounds"] = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
    q["segments"] = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    q["bootstrap_rounds"] = conn.execute("SELECT COUNT(*) FROM rounds WHERE is_bootstrap=1").fetchone()[0]
    q["not_2dp"] = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE ABS(multiplier*100 - ROUND(multiplier*100)) > 1e-6").fetchone()[0]
    q["ambiguous_alignments"] = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE json_extract(raw_data,'$.ambiguous') = 1").fetchone()[0]
    q["duplicate_round_ids"] = conn.execute(
        "SELECT COUNT(*) FROM (SELECT round_id FROM rounds WHERE round_id IS NOT NULL"
        " GROUP BY round_id HAVING COUNT(*)>1)").fetchone()[0]
    # Every scored live prediction must have been made before its target was observed.
    q["predictions_after_target"] = conn.execute(
        "SELECT COUNT(*) FROM predictions p JOIN rounds r ON r.id=p.target_round_pk"
        " WHERE p.run_id=0 AND p.predicted_at >= r.observed_at").fetchone()[0]
    q["live_predictions"] = dict(conn.execute(
        "SELECT status, COUNT(*) FROM predictions WHERE run_id=0 GROUP BY status").fetchall())
    seglens = [r[0] for r in conn.execute("SELECT COUNT(*) FROM rounds GROUP BY segment_id")]
    q["segment_lengths"] = seglens
    return q


def live_scoreboard(conn):
    return conn.execute(
        "SELECT model_name, COUNT(*) n, AVG(correct) acc, AVG(log_loss) ll, AVG(brier) brier"
        " FROM predictions WHERE run_id=0 AND status='resolved' GROUP BY model_name ORDER BY ll"
    ).fetchall()


def main():
    conn = db.connect()
    q = quality_checks(conn)
    print("=== DATA QUALITY ===")
    for k, v in q.items():
        print(f"{k:28} {v}")
    if q["predictions_after_target"]:
        print("!! Some predictions were stamped after their target round: investigate before trusting live scores.")
    df = db.load_rounds(conn)
    if df.empty:
        print("\nNo rounds yet. Run: python -m prediction_lab collect")
        return
    x, seg = df["multiplier"].to_numpy(), df["segment_id"].to_numpy()
    print(f"\n=== DISTRIBUTION (n={len(x)}) ===")
    print(f"median {np.median(x):.2f}x   mean {x.mean():.2f}x   max {x.max():.2f}x")
    d = stats.distribution_vs_theory(x)
    for lab, o, e in zip(LABELS, d["observed"], d["expected"]):
        print(f"{lab:12} {o:6} ({o / len(x):6.1%})   expected@RTP97 {e / len(x):6.1%}")
    print(f"chi-square vs RTP97 model: p={d['p']:.3f}")
    print("implied RTP  m·P(X≥m):  " + "  ".join(
        f"{m}x→{v['estimate']:.3f}±{v['se']:.3f}" for m, v in stats.implied_rtp(x).items()))

    print("\n=== INDEPENDENCE TESTS (small p = evidence of dependence) ===")
    if len(x) >= 50:
        lb = stats.ljung_box(np.log(x), lags=min(20, len(x) // 5))
        print(f"Ljung-Box on log(x): p={lb['p']:.3f}   max|acf|={max(map(abs, lb['acf'])):.3f}"
              f" (95% band ±{lb['acf_band95']:.3f})")
        rt = stats.runs_test(x)
        print(f"Runs test above/below median: runs={rt['runs']} expected={rt['expected']:.1f} p={rt['p']:.3f}")
        tt = stats.transition_test(x, seg)
        print(f"Class transition chi-square: p={tt['p']:.3f}"
              + ("  (low expected counts; unreliable)" if tt.get("min_expected", 5) < 5 else ""))
    else:
        print("Need ≥50 rounds.")
    if len(x) < 500:
        print("\nNOTE: fewer than 500 rounds. Treat every number above as descriptive only.")

    rows = live_scoreboard(conn)
    if rows:
        print("\n=== LIVE PREDICTIONS (scored on rounds after the prediction was logged) ===")
        for r in rows:
            print(f"{r['model_name']:16} n={r['n']:5}  acc={r['acc']:.1%}  logloss={r['ll']:.4f}  brier={r['brier']:.4f}")
