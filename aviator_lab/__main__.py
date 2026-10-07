"""Command line: python -m aviator_lab <command>"""
import argparse
import asyncio
import json
import subprocess
import sys

from . import config


def cmd_probe(a):
    from .collector import probe
    asyncio.run(probe(a.url))


def cmd_collect(a):
    from .collector import run
    try:
        asyncio.run(run(selector=a.selector, url=a.url, live=not a.no_predict, order=a.order))
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_evaluate(a):
    from . import database as db
    from .evaluation import format_leaderboard, walk_forward
    from .simulation import generate_random_rounds, generate_signal_rounds

    conn = db.connect()
    if a.data == "real":
        df = db.load_rounds(conn)
        x, seg = df["multiplier"].to_numpy(), df["segment_id"].to_numpy()
    elif a.data == "synthetic-null":
        x, seg = generate_random_rounds(a.n, seed=a.seed), None
    else:
        x, seg = generate_signal_rounds(a.n, seed=a.seed), None
    print(f"Dataset: {a.data}  rounds={len(x)}  window={a.window or 'expanding'}  "
          f"retrain_every={a.retrain_every}  {'FINAL HOLDOUT' if a.final else f'dev (holdout {a.holdout:.0%} excluded)'}")
    if a.final:
        print("WARNING: you are looking at the final holdout. Do this rarely, and never tune afterwards.")
    res = walk_forward(x, seg, initial_train=a.initial_train, window=a.window, holdout=a.holdout,
                       final=a.final, retrain_every=a.retrain_every, progress=True)
    print(format_leaderboard(res))
    dataset = a.data + ("-final" if a.final else "")
    run_id = db.save_model_run(conn, dataset, res.config | {"n": len(x), "seed": a.seed},
                               len(x), len(res.y), res.to_results())
    if a.data == "real":  # keep every backtest prediction for reconstruction
        from .metrics import per_row_brier, per_row_log_loss
        pks = df["id"].to_numpy()
        now = db.utcnow()
        rows = []
        for m in res.models:
            P = res.probs[m.name]
            ll, br = per_row_log_loss(res.y, P), per_row_brier(res.y, P)
            for i, t in enumerate(res.eval_index):
                rows.append((run_id, m.name, now, int(pks[t - 1]), int(pks[t]), *map(float, P[i]),
                             int(P[i].argmax()), float(x[t]), int(res.y[i]),
                             int(P[i].argmax() == res.y[i]), float(ll[i]), float(br[i]), now))
        conn.executemany(
            "INSERT INTO predictions (run_id, model_name, predicted_at, base_round_pk, target_round_pk,"
            " p0,p1,p2,p3,p4, predicted_class, status, actual_multiplier, actual_class, correct,"
            " log_loss, brier, resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,'resolved',?,?,?,?,?,?)", rows)
        conn.commit()
    print(f"Saved as model_run #{run_id}.")
    if len(x) < 1000:
        print("NOTE: <1,000 rounds. Differences between models are expected to be noise at this size.")


def cmd_simulate(a):
    from . import database as db
    from .evaluation import format_leaderboard
    from .simulation import run_sanity_checks
    out = run_sanity_checks(n=a.n, seed=a.seed, retrain_every=a.retrain_every, progress=True)
    conn = db.connect()
    for kind in ("null", "signal"):
        res = out[kind]["result"]
        print(f"\n=== synthetic {kind} (n={a.n}, seed={a.seed}) ===")
        print(format_leaderboard(res))
        db.save_model_run(conn, f"synthetic-{kind}", res.config | {"n": a.n, "seed": a.seed},
                          a.n, len(res.y), res.to_results())
    print("\nNULL check   (no learned model may beat baseline on i.i.d. data):",
          "PASS" if out["null_pass"] else f"FAIL -> {out['null']['learned_winners']} (investigate leakage)")
    print("SIGNAL check (pipeline must detect a planted pattern):           ",
          "PASS" if out["signal_pass"] else "FAIL -> tests lack power at this n/strength")
    print("(At a 5% significance level the null check fails by chance ~5% of the time; re-run with another --seed.)")


def cmd_report(a):
    from .report import main
    main()


def cmd_verify(a):
    from .fairness import verify_round
    print(json.dumps(verify_round(a.server_seed, a.client_seed, a.multiplier, a.hash, a.server_seed_hash), indent=2))


def cmd_dashboard(a):
    app = config.BASE_DIR / "dashboard" / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="aviator_lab", description="Aviator Research Lab (observe-only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="find the history strip on the page")
    s.add_argument("--url", default=config.AVIATOR_URL)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("collect", help="observe rounds and log live predictions")
    s.add_argument("--url", default=config.AVIATOR_URL)
    s.add_argument("--selector", default=config.HISTORY_ITEM_SELECTOR)
    s.add_argument("--order", choices=["newest_first", "oldest_first"], default="newest_first")
    s.add_argument("--no-predict", action="store_true")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("evaluate", help="walk-forward backtest of all models")
    s.add_argument("--data", choices=["real", "synthetic-null", "synthetic-signal"], default="real")
    s.add_argument("--n", type=int, default=5000, help="synthetic rounds")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--initial-train", type=int, default=100)
    s.add_argument("--window", type=int, default=None, help="rolling window (default expanding)")
    s.add_argument("--retrain-every", type=int, default=25)
    s.add_argument("--holdout", type=float, default=0.2)
    s.add_argument("--final", action="store_true", help="evaluate on the final holdout")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("simulate", help="null + planted-signal pipeline sanity checks")
    s.add_argument("--n", type=int, default=5000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--retrain-every", type=int, default=50)
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("report", help="data quality, distribution, independence tests")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("verify-round", help="test provably-fair hypotheses on one round")
    s.add_argument("--server-seed", required=True)
    s.add_argument("--client-seed", action="append", required=True)
    s.add_argument("--multiplier", type=float, required=True)
    s.add_argument("--hash")
    s.add_argument("--server-seed-hash")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("dashboard", help="open the Streamlit dashboard")
    s.set_defaults(fn=cmd_dashboard)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
