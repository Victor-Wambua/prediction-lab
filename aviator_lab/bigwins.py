"""Big-multiplier analysis: do 10x / 50x / 100x / 1000x+ rounds follow a pattern?

Questions, each with a test whose null hypothesis is "pure chance":
1. Frequency  - is the rate what a 97% RTP crash game implies (P(X>=m) = 0.97/m)?
2. "Is it due"- does the chance of a big round rise with rounds since the last one?
                (a fair game is memoryless: the rate is flat across drought lengths)
3. Clustering - is a big round more likely right after another big round?
4. Time of day- does the rate differ by time of day (Nairobi time)?

Gaps and droughts are only measured inside a segment (a run with no collection
gaps), because rounds missed during a gap are unknown. Bootstrap rounds are
excluded from time-of-day analysis since their real times are unknown.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import holm

THRESHOLDS = (10.0, 50.0, 100.0, 1000.0)
RTP = 0.97
MIN_EVENTS_FOR_TESTS = 20
NAIROBI_OFFSET_H = 3
TOD_BLOCKS = ["00–04", "04–08", "08–12", "12–16", "16–20", "20–24"]


def expected_rate(m, rtp=RTP):
    return rtp / m


def rounds_since(event: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """For each round t: rounds since the most recent event BEFORE t in the same
    segment (1 = the previous round was an event). NaN if none seen yet."""
    out = np.full(len(event), np.nan)
    last, seg = None, None
    for i, (e, s) in enumerate(zip(event, segment)):
        if s != seg:
            last, seg = None, s
        if last is not None:
            out[i] = i - last
        if e:
            last = i
    return out


def gaps(event: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Rounds between consecutive events within the same segment."""
    idx = np.flatnonzero(event)
    if len(idx) < 2:
        return np.array([], dtype=int)
    same = segment[idx[1:]] == segment[idx[:-1]]
    return (idx[1:] - idx[:-1])[same]


@dataclass
class ThresholdResult:
    threshold: float
    n_rounds: int
    n_events: int
    rate: float
    expected: float
    rate_ci: tuple
    freq_p: float
    current_drought: int | None      # rounds since last event in latest segment
    drought_is_lower_bound: bool     # no event yet in this segment
    drought_chance: float | None     # P(a drought at least this long) by chance
    gaps: np.ndarray
    hazard: pd.DataFrame             # rate by drought-length bin
    due_p: float | None              # long-vs-short drought Fisher test
    cluster_p: float | None
    cluster_rates: tuple | None      # (rate after a recent event, rate otherwise)
    tod: pd.DataFrame | None
    tod_p: float | None
    enough: bool


def analyse_threshold(x, segment, observed_at=None, is_bootstrap=None, m=100.0,
                      cluster_window=10) -> ThresholdResult:
    x = np.asarray(x, float)
    segment = np.asarray(segment)
    n = len(x)
    ev = x >= m
    k = int(ev.sum())
    p0 = expected_rate(m)
    bt = st.binomtest(k, n, p0) if n else None
    ci = tuple(bt.proportion_ci(0.95)) if bt else (np.nan, np.nan)
    enough = k >= MIN_EVENTS_FOR_TESTS

    # Current drought in the latest segment.
    last_seg = segment[-1] if n else None
    in_last = np.flatnonzero(segment == last_seg) if n else np.array([], int)
    ev_last = np.flatnonzero(ev[in_last]) if n else np.array([], int)
    if n == 0:
        drought, lower = None, False
    elif len(ev_last):
        drought, lower = int(len(in_last) - 1 - ev_last[-1]), False
    else:
        drought, lower = int(len(in_last)), True
    chance = float((1 - p0) ** drought) if drought is not None else None

    # Hazard: event rate as a function of rounds since the previous event.
    since = rounds_since(ev, segment)
    g = 1.0 / p0
    edges = [0, 0.25 * g, 0.5 * g, g, 2 * g, np.inf]
    labels = [f"≤{int(edges[1])}", f"{int(edges[1]) + 1}–{int(edges[2])}",
              f"{int(edges[2]) + 1}–{int(edges[3])}", f"{int(edges[3]) + 1}–{int(edges[4])}",
              f">{int(edges[4])}"]
    known = ~np.isnan(since)
    bins = pd.cut(since[known], edges, labels=labels, right=True, include_lowest=True)
    hz = pd.DataFrame({"bin": bins, "event": ev[known]}).groupby("bin", observed=False)["event"].agg(
        rounds="size", events="sum")
    hz["rate"] = hz.events / hz.rounds.where(hz.rounds > 0)
    hz["expected"] = p0
    hz = hz.reset_index().rename(columns={"bin": "rounds_since_last"})

    due_p = cluster_p = tod_p = None
    cluster_rates = tod = None
    if known.sum() and enough:
        long_ = since[known] > g * np.log(2)  # longer than the median chance gap
        table = [[int((ev[known] & long_).sum()), int((~ev[known] & long_).sum())],
                 [int((ev[known] & ~long_).sum()), int((~ev[known] & ~long_).sum())]]
        due_p = float(st.fisher_exact(table)[1])
        recent = since[known] <= cluster_window
        a = ev[known][recent]
        b = ev[known][~recent]
        if len(a) and len(b):
            cluster_rates = (float(a.mean()), float(b.mean()))
            cluster_p = float(st.fisher_exact([[a.sum(), len(a) - a.sum()],
                                               [b.sum(), len(b) - b.sum()]])[1])

    if observed_at is not None and enough:
        ts = pd.to_datetime(pd.Series(observed_at), utc=True, format="ISO8601")
        live = ~np.asarray(is_bootstrap, bool) if is_bootstrap is not None else np.ones(n, bool)
        hours = ((ts.dt.hour + NAIROBI_OFFSET_H) % 24).to_numpy()[live]
        block = hours // 4
        e = ev[live]
        tod = pd.DataFrame({"block": block, "event": e}).groupby("block")["event"].agg(
            rounds="size", events="sum")
        tod = tod.reindex(range(6), fill_value=0)
        tod["rate"] = tod.events / tod.rounds.where(tod.rounds > 0)
        tod.index = [TOD_BLOCKS[i] for i in tod.index]
        populated = tod[tod.rounds > 0]
        if len(populated) >= 2:
            tab = np.vstack([populated.events, populated.rounds - populated.events])
            if tab[0].sum() > 0:
                tod_p = float(st.chi2_contingency(tab)[1])

    return ThresholdResult(m, n, k, k / n if n else np.nan, p0, ci, float(bt.pvalue) if bt else np.nan,
                           drought, lower, chance, gaps(ev, segment), hz, due_p, cluster_p,
                           cluster_rates, tod, tod_p, enough)


def analyse(x, segment, observed_at=None, is_bootstrap=None, thresholds=THRESHOLDS):
    res = {m: analyse_threshold(x, segment, observed_at, is_bootstrap, m) for m in thresholds}
    # Many tests are run (3 pattern tests × 4 thresholds), so correct for multiplicity.
    pv = {}
    for m, r in res.items():
        for name, p in (("due", r.due_p), ("cluster", r.cluster_p), ("time_of_day", r.tod_p)):
            if p is not None:
                pv[(m, name)] = p
    return res, holm(pv) if pv else {}
