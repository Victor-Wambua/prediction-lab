"""Aviator Research Lab dashboard. Run: python -m aviator_lab dashboard"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aviator_lab import bigwins, database as db, stats, targets, tiers  # noqa: E402
from aviator_lab.classes import LABELS, theoretical_probs  # noqa: E402
from aviator_lab.metrics import calibration_table  # noqa: E402
from aviator_lab.report import quality_checks  # noqa: E402

st.set_page_config(page_title="Aviator Research Lab", layout="wide")
st.title("Aviator Research Lab")
st.caption("Observe-only research. Probabilities are model estimates, not certainty. "
           "No result here is evidence of a winning strategy unless it beats the baseline "
           "on unseen data with statistical significance.")


def tier_panel(conn):
    st.subheader("Next round: risk tiers")
    pend = pd.read_sql_query(
        "SELECT p.model_name, p.p_low, p.p_mid, p.p_high, p.predicted_at, r.multiplier AS after_x"
        " FROM tier_predictions p JOIN rounds r ON r.id=p.base_round_pk"
        " WHERE p.status='pending' AND p.base_round_pk=(SELECT MAX(base_round_pk) FROM tier_predictions)",
        conn)
    if pend.empty:
        st.info("Waiting for the collector to log a tier forecast (one appears after the next round ends).")
        return
    ref = pend.set_index("model_name").loc["global_freq"]
    st.caption(f"Forecast for the round after **{ref.after_x:.2f}x**, logged "
               f"{ref.predicted_at[11:19]} UTC, **awaiting result**. Headline numbers = global_freq "
               "(the reference baseline). These are estimated chances, not certainty.")
    cols = st.columns(3)
    for col, label, key in zip(cols, tiers.TIER_LABELS, ["p_low", "p_mid", "p_high"]):
        col.metric(label, f"{ref[key]:.0%}")
    t = pend.set_index("model_name")[["p_low", "p_mid", "p_high"]]
    t.columns = tiers.TIER_SHORT
    with st.expander("All tier models"):
        st.dataframe(t.style.format("{:.1%}"), width="stretch")

    hist = pd.read_sql_query(
        "SELECT resolved_at, p_low, p_mid, p_high, actual_multiplier, actual_tier, correct, log_loss"
        " FROM tier_predictions WHERE status='resolved' AND model_name='global_freq'"
        " ORDER BY id DESC LIMIT 200", conn)
    if not hist.empty:
        last = hist.iloc[0]
        st.write(f"Last round: **{last.actual_multiplier:.2f}x → {tiers.TIER_SHORT[int(last.actual_tier)]}**"
                 f" (forecast was Low {last.p_low:.0%} / Middle {last.p_mid:.0%} / High {last.p_high:.0%})")
        score = pd.read_sql_query(
            "SELECT model_name AS model, COUNT(*) n, AVG(correct) AS hit_rate, AVG(log_loss) AS log_loss"
            " FROM tier_predictions WHERE status='resolved' GROUP BY model_name ORDER BY log_loss", conn)
        actual = np.bincount(hist.actual_tier.astype(int), minlength=3) / len(hist)
        with st.expander(f"Tier forecast track record ({int(score.n.max())} rounds scored)"):
            st.dataframe(score.style.format({"hit_rate": "{:.1%}", "log_loss": "{:.4f}"}),
                         width="stretch", hide_index=True)
            st.caption("Actual tier mix over the last %d rounds: " % len(hist) + ", ".join(
                f"{s} {a:.0%}" for s, a in zip(tiers.TIER_SHORT, actual))
                + ". hit_rate = how often the most likely tier happened.")
            recent = hist.head(15).copy()
            recent["result"] = recent.actual_multiplier.map("{:.2f}x".format)
            recent["tier"] = recent.actual_tier.map(lambda i: tiers.TIER_SHORT[int(i)])
            recent["forecast L/M/H"] = recent.apply(
                lambda r: f"{r.p_low:.0%} / {r.p_mid:.0%} / {r.p_high:.0%}", axis=1)
            st.dataframe(recent[["result", "tier", "forecast L/M/H"]], width="stretch", hide_index=True)


GREEN, RED = "#1a7f37", "#cf222e"
TARGET_WARMUP = 200  # rounds of history before a target prediction counts toward "beating chance"


def target_panel(conn):
    level = st.radio("Prediction level", targets.LEVELS, horizontal=True, key="target_level",
                     format_func=lambda v: f"{v:.0%} chance",
                     help="Higher target = rarer hit. Each level's hit rate should end up near "
                          "its own % if past rounds don't predict the next one.")
    pend = conn.execute(
        "SELECT p.target, p.predicted_at, r.multiplier AS after_x FROM target_predictions p"
        " JOIN rounds r ON r.id=p.base_round_pk WHERE p.status='pending' AND p.level=?"
        " ORDER BY p.base_round_pk DESC LIMIT 1", (level,)).fetchone()
    if pend is None:
        st.info("Waiting for the collector to log the next prediction (appears after the next round ends).")
    else:
        st.markdown(
            f"<div style='font-size:1rem;opacity:.75'>Next round prediction</div>"
            f"<div style='font-size:3.2rem;font-weight:700;line-height:1.1'>at least "
            f"{pend['target']:.2f}x</div>", unsafe_allow_html=True)
        st.caption(f"{level:.0%} of past rounds reached this (fair-game value: "
                   f"{targets.fair_game_target(level):.2f}x). Logged {pend['predicted_at'][11:19]} UTC "
                   f"after {pend['after_x']:.2f}x, awaiting result.")

    hist = pd.read_sql_query(
        "SELECT p.target, p.actual_multiplier, p.hit,"
        " (SELECT COUNT(*) FROM rounds r2 WHERE r2.id <= p.base_round_pk) AS n_history"
        " FROM target_predictions p WHERE p.status='resolved' AND p.level=? ORDER BY p.id",
        conn, params=(level,))
    if hist.empty:
        return
    last = hist.tail(40)
    chips = "".join(
        f"<span title='predicted ≥{r.target:.2f}x' style='display:inline-block;margin:2px;"
        f"padding:3px 7px;border-radius:6px;color:#fff;font-size:.85rem;"
        f"background:{GREEN if r.hit else RED}'>{r.actual_multiplier:.2f}x</span>"
        for r in last.itertuples())
    st.markdown(f"<div style='margin:.25rem 0 .5rem'>{chips}</div>", unsafe_allow_html=True)
    st.caption(f"Last {len(last)} outcomes, oldest → newest. Green = reached the predicted "
               "multiplier, red = crashed below it. Hover for the prediction.")

    # Targets computed from a short history are noisy (e.g. a 10% target from 25
    # rounds), which inflates or deflates hit rates. Judge only after warm-up, and
    # correct for checking several levels at once.
    scored = hist[hist.n_history >= TARGET_WARMUP]
    n, hits = len(scored), int(scored.hit.sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correct overall", f"{hist.hit.mean():.1%}", help=f"{int(hist.hit.sum())} of {len(hist)} rounds")
    c2.metric("Correct, last 50", f"{hist.hit.tail(50).mean():.1%}")
    c3.metric("Expected by chance", f"{level:.0%}")
    if n < 100:
        c4.metric("Beating chance?", "Too early",
                  help=f"Judged on predictions made with ≥{TARGET_WARMUP} rounds of history; have {n}.")
    else:
        p_adj = min(1.0, stats_binom(hits, n, level) * len(targets.LEVELS))
        c4.metric("Beating chance?", "Yes, investigate" if p_adj < 0.05 else "No",
                  help=f"After warm-up: {hits}/{n} = {hits / n:.1%}. One-sided binomial p, "
                       f"Bonferroni-corrected for {len(targets.LEVELS)} levels = {p_adj:.3f}.")
    cum = pd.DataFrame({"% correct so far": hist.hit.expanding().mean() * 100,
                        "expected by chance": level * 100})
    cum.index = np.arange(1, len(hist) + 1)
    st.line_chart(cum, height=200)


def stats_binom(hits, n, p):
    from scipy.stats import binomtest
    return binomtest(hits, n, p, alternative="greater").pvalue


@st.fragment(run_every=5)
def live_panel():
    conn = db.connect()
    target_panel(conn)
    st.divider()
    rounds = db.load_rounds(conn)
    q = quality_checks(conn)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds collected", f"{len(rounds):,}")
    c2.metric("Latest round", f"{rounds.multiplier.iloc[-1]:.2f}x" if len(rounds) else "—")
    c3.metric("Segments (gaps + 1)", q["segments"])
    c4.metric("Live predictions scored", q["live_predictions"].get("resolved", 0))
    if q["predictions_after_target"]:
        st.error(f"{q['predictions_after_target']} live predictions were stamped after their target round. "
                 "Live scores are not trustworthy until this is explained.")

    tier_panel(conn)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Next round: estimated probabilities (5 research classes)")
        st.caption("Logged before the round happens; scored automatically once it ends.")
        pend = pd.read_sql_query(
            "SELECT model_name, p0,p1,p2,p3,p4 FROM predictions WHERE run_id=0 AND status='pending'"
            " AND base_round_pk=(SELECT MAX(base_round_pk) FROM predictions WHERE run_id=0)", conn)
        if pend.empty:
            st.info("No pending prediction. Start the collector: python -m aviator_lab collect")
        else:
            t = pend.set_index("model_name")[db.PROB_COLS]
            t.columns = LABELS
            st.dataframe(t.style.format("{:.1%}"), width="stretch")
    with right:
        st.subheader("Recent results")
        recent = rounds.tail(30).iloc[::-1]
        st.write("  ".join(f"`{v:.2f}x`" for v in recent.multiplier))

    st.subheader("Live performance (predictions logged before the round happened)")
    live = pd.read_sql_query(
        "SELECT p.model_name, p.correct, p.log_loss, p.brier, p.resolved_at, p.id FROM predictions p"
        " WHERE p.run_id=0 AND p.status='resolved' ORDER BY p.id", conn)
    if live.empty:
        st.write("No scored live predictions yet.")
    else:
        rows = []
        for name, g in live.groupby("model_name"):
            rows.append({"model": name, "n": len(g),
                         "acc last 100": g.correct.tail(100).mean(),
                         "acc last 500": g.correct.tail(500).mean(),
                         "acc all": g.correct.mean(),
                         "log loss all": g.log_loss.mean(), "brier all": g.brier.mean()})
        lb = pd.DataFrame(rows).sort_values("log loss all")
        st.dataframe(lb.style.format({"acc last 100": "{:.1%}", "acc last 500": "{:.1%}",
                                      "acc all": "{:.1%}", "log loss all": "{:.4f}",
                                      "brier all": "{:.4f}"}), width="stretch", hide_index=True)
        roll = {name: g.log_loss.rolling(100, min_periods=20).mean().reset_index(drop=True)
                for name, g in live.groupby("model_name")}
        st.caption("Rolling log loss (100 rounds, lower is better)")
        st.line_chart(pd.DataFrame(roll))


live_panel()


@st.fragment(run_every=30)
def big_multipliers_panel():
    st.header("Big multipliers")
    st.caption("Do 10x / 50x / 100x / 1000x+ rounds follow a pattern you could time? Each test's "
               "default assumption is pure chance; it only flags a pattern if the data clearly "
               "contradicts that (p < 0.05 after correcting for running many tests).")
    conn = db.connect()
    r = db.load_rounds(conn)
    if r.empty:
        st.info("No rounds yet.")
        return
    res, adj = bigwins.analyse(r.multiplier.to_numpy(), r.segment_id.to_numpy(),
                               r.observed_at.to_numpy(), r.is_bootstrap.to_numpy())

    summary = []
    for m, t in res.items():
        drought = "—" if t.current_drought is None else (
            f"{t.current_drought}+" if t.drought_is_lower_bound else str(t.current_drought))
        summary.append({
            "threshold": f"{m:g}x+",
            "seen": t.n_events,
            "observed rate": f"{t.rate:.2%}  (1 in {1 / t.rate:,.0f})" if t.n_events else "0",
            "fair-game rate": f"{t.expected:.2%}  (1 in {1 / t.expected:,.0f})",
            "95% range for true rate": f"{t.rate_ci[0]:.2%} – {t.rate_ci[1]:.2%}",
            "rounds since last": drought,
            "chance of a drought this long": f"{t.drought_chance:.0%}" if t.drought_chance is not None else "—",
        })
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    st.caption(f"Based on {len(r):,} rounds. 'rounds since last' with + means none seen yet in the "
               "current collection segment. A long drought is not a sign one is coming: in a fair "
               "game the next round's chance stays at the fair-game rate.")

    m = st.radio("Look closer at", list(res), format_func=lambda v: f"{v:g}x+", horizontal=True, index=2)
    t = res[m]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Seen ≥{m:g}x", t.n_events, help="Count in all collected rounds")
    c2.metric("Rounds since last", "—" if t.current_drought is None else
              (f"{t.current_drought}+" if t.drought_is_lower_bound else t.current_drought))
    c3.metric(f"Chance next round is ≥{m:g}x (fair game)", f"{t.expected:.2%}",
              help="The same every round in a fair game, whatever happened before.")

    if not t.enough:
        more = (bigwins.MIN_EVENTS_FOR_TESTS - t.n_events) / t.expected
        st.info(f"Pattern tests need at least {bigwins.MIN_EVENTS_FOR_TESTS} rounds of ≥{m:g}x; "
                f"have {t.n_events}. At the fair-game rate that's roughly {more:,.0f} more rounds.")
    else:
        rows = []
        def verdict(p):
            return "—" if p is None else ("Pattern detected: investigate" if p < 0.05 else "No evidence of a pattern")
        due = adj.get((m, "due"))
        rows.append({"question": "More likely after a long drought? (is it 'due')",
                     "p (corrected)": due, "result": verdict(due)})
        cl = adj.get((m, "cluster"))
        detail = (f" (rate {t.cluster_rates[0]:.2%} within 10 rounds of one vs {t.cluster_rates[1]:.2%} otherwise)"
                  if t.cluster_rates else "")
        rows.append({"question": "More likely right after another big one? (clustering)" + detail,
                     "p (corrected)": cl, "result": verdict(cl)})
        tod = adj.get((m, "time_of_day"))
        rows.append({"question": "Depends on time of day (Nairobi)?", "p (corrected)": tod, "result": verdict(tod)})
        st.dataframe(pd.DataFrame(rows).style.format({"p (corrected)": lambda v: "—" if v is None else f"{v:.3f}"}),
                     width="stretch", hide_index=True)

    st.caption(f"Rate of ≥{m:g}x by rounds since the previous one. Flat bars at the fair-game "
               "line = memoryless (no 'due' effect). Bins with few rounds are noisy.")
    hz = t.hazard.set_index("rounds_since_last")[["rate", "expected"]]
    hz.columns = ["observed rate", "fair-game rate"]
    st.bar_chart(hz, stack=False)
    counts = t.hazard.set_index("rounds_since_last")[["rounds", "events"]].astype(int).T
    counts.columns = [str(c) for c in counts.columns]
    st.dataframe(counts, width="stretch")

    if len(t.gaps):
        st.caption(f"Gaps between ≥{m:g}x rounds: median {np.median(t.gaps):.0f} observed vs "
                   f"{np.log(2) / t.expected:.0f} expected by chance; longest {t.gaps.max()}.")
    if t.tod is not None:
        with st.expander("By time of day (Nairobi, live-observed rounds only)"):
            st.dataframe(t.tod.style.format({"rate": "{:.2%}"}), width="stretch")

    big = r[r.multiplier >= 10].copy()
    if len(big):
        big["since previous ≥10x"] = big.index.to_series().diff()
        big.loc[big.segment_id.ne(big.segment_id.shift()), "since previous ≥10x"] = np.nan
        ts = pd.to_datetime(big.observed_at, utc=True, format="ISO8601") + pd.Timedelta(hours=3)
        big["seen at (Nairobi)"] = ts.dt.strftime("%d %b %H:%M").where(big.is_bootstrap == 0, "from history strip")
        big["multiplier"] = big.multiplier.map("{:,.2f}x".format)
        with st.expander(f"Recent rounds ≥10x ({len(big)} total)"):
            st.dataframe(big.iloc[::-1].head(30)[["multiplier", "seen at (Nairobi)", "since previous ≥10x"]],
                         width="stretch", hide_index=True)


big_multipliers_panel()

conn = db.connect()
rounds = db.load_rounds(conn)

st.header("Distribution")
if len(rounds):
    x = rounds.multiplier.to_numpy()
    d = stats.distribution_vs_theory(x)
    dist = pd.DataFrame({"observed": np.array(d["observed"]) / len(x),
                         "RTP 97% model (assumption)": theoretical_probs(0.97)}, index=LABELS)
    st.bar_chart(dist, stack=False)
    st.caption(f"n={len(x)}; chi-square vs RTP97 p={d['p']:.3f}. log10(multiplier) histogram:")
    counts, edges = np.histogram(np.log10(x), bins=40)
    st.bar_chart(pd.DataFrame({"rounds": counts}, index=np.round(10 ** edges[:-1], 2)))

st.header("Latest backtest (walk-forward)")
runs = pd.read_sql_query("SELECT * FROM model_runs ORDER BY id DESC LIMIT 20", conn)
if runs.empty:
    st.write("No backtests yet. Run: python -m aviator_lab evaluate  (or simulate)")
else:
    choice = st.selectbox("Run", runs.id, format_func=lambda i: (
        lambda r: f"#{i} {r.dataset} n={r.n_rounds} scored={r.n_scored} {r.created_at[:19]}")(
        runs.set_index("id").loc[i]))
    r = runs.set_index("id").loc[choice]
    res = json.loads(r.results_json)
    lb = pd.DataFrame(res["leaderboard"]).set_index("model")
    sig = pd.DataFrame(res["significance"]).T
    show = lb[["accuracy", "balanced_accuracy", "log_loss", "train_log_loss", "brier", "ece"]].join(
        sig[["mean_diff", "p_holm", "significant"]], how="left").sort_values("log_loss")
    st.dataframe(show, width="stretch")
    st.caption("mean_diff = log loss minus global_freq baseline (negative = better). "
               "Large gap between train_log_loss and log_loss suggests overfitting.")

    if r.dataset.startswith("real"):
        bt = pd.read_sql_query(
            "SELECT model_name, actual_class, p0,p1,p2,p3,p4 FROM predictions WHERE run_id=?",
            conn, params=(int(choice),))
        model = st.selectbox("Calibration for model", sorted(bt.model_name.unique()))
        g = bt[bt.model_name == model]
        cal = calibration_table(g.actual_class.to_numpy(), g[db.PROB_COLS].to_numpy())
        cal = cal.set_index("mean_predicted")[["observed"]]
        cal["perfect"] = cal.index
        st.line_chart(cal)
        st.caption("Calibration: observed frequency vs predicted probability (closer to 'perfect' is better).")

st.header("Data quality")
st.json(quality_checks(conn))
