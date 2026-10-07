"""Feature engineering.

Contract (enforced by tests/test_features.py): row t of the feature matrix is a
function of x[0..t-1] of the same segment ONLY. Changing x[t] or anything after
it must not change row t. Features never cross segment boundaries.

Statistics are computed on log(multiplier) to tame the heavy right tail.
"""
import numpy as np
import pandas as pd

from .classes import N_CLASSES, to_class

LAGS = 10
WINDOWS = (5, 10, 25, 50)
PCT_WINDOWS = (10, 50)
UNDER = (1.2, 1.5, 2.0)
OVER = (3.0, 5.0, 10.0)
RECENCY_CAP = 200


def _run_length(cond: np.ndarray) -> np.ndarray:
    """Length of the run of True ending at each index (inclusive)."""
    out = np.zeros(len(cond))
    run = 0
    for i, c in enumerate(cond):
        run = run + 1 if c else 0
        out[i] = run
    return out


def _since(event: np.ndarray, cap: int) -> np.ndarray:
    """Rounds elapsed since the most recent event, inclusive (0 if event at i)."""
    out = np.full(len(event), np.nan)
    last = None
    for i, e in enumerate(event):
        if e:
            last = i
        if last is not None:
            out[i] = min(i - last, cap)
    return out


def _segment_features(x: np.ndarray) -> pd.DataFrame:
    lx = pd.Series(np.log(x))
    cls = pd.Series(to_class(x).astype(float))
    f = {}

    # "inclusive" stats describe x[0..i]; shifting by 1 makes row t describe x[0..t-1].
    for k in range(1, LAGS + 1):
        f[f"lag_{k}"] = lx.shift(k)
    for w in WINDOWS:
        r = lx.rolling(w, min_periods=w)
        f[f"mean_{w}"] = r.mean().shift(1)
        f[f"median_{w}"] = r.median().shift(1)
        if w >= 10:
            f[f"std_{w}"] = r.std().shift(1)
            q75, q25 = r.quantile(0.75).shift(1), r.quantile(0.25).shift(1)
            f[f"iqr_{w}"] = q75 - q25
    for w in PCT_WINDOWS:
        for t in UNDER:
            f[f"pct_under_{t}_{w}"] = pd.Series(x < t, dtype=float).rolling(w, min_periods=w).mean().shift(1)
        for t in OVER:
            f[f"pct_over_{t}_{w}"] = pd.Series(x >= t, dtype=float).rolling(w, min_periods=w).mean().shift(1)

    f["streak_under_1.5"] = pd.Series(_run_length(x < 1.5)).shift(1)
    f["streak_under_2"] = pd.Series(_run_length(x < 2.0)).shift(1)
    f["streak_over_3"] = pd.Series(_run_length(x >= 3.0)).shift(1)
    f["streak_over_5"] = pd.Series(_run_length(x >= 5.0)).shift(1)
    for t in OVER:
        f[f"since_{t}x"] = pd.Series(_since(x >= t, RECENCY_CAP)).shift(1)

    for k in (1, 2):
        prev = cls.shift(k)
        for c in range(N_CLASSES):
            f[f"prev{k}_is_c{c}"] = (prev == c).astype(float).where(prev.notna())

    return pd.DataFrame(f)


def build_features(x, segment=None) -> pd.DataFrame:
    """Feature matrix with one row per round (row t = what was known before round t)."""
    x = np.asarray(x, dtype=float)
    if segment is None:
        segment = np.zeros(len(x), dtype=int)
    segment = np.asarray(segment)
    parts = []
    # Preserve the input order; rows of each segment must already be contiguous.
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    ends = np.r_[starts[1:], len(x)]
    for s, e in zip(starts, ends):
        parts.append(_segment_features(x[s:e]))
    if not parts:
        return _segment_features(np.array([1.0])).iloc[:0]
    out = pd.concat(parts, ignore_index=True)
    return out


def features_for_next(x_segment) -> pd.DataFrame:
    """Single feature row for the round AFTER the given segment history.

    The appended placeholder is never read (row t only depends on x[<t]).
    """
    x = np.append(np.asarray(x_segment, dtype=float), 1.0)
    return _segment_features(x).iloc[[-1]].reset_index(drop=True)


FEATURE_NAMES = list(_segment_features(np.ones(3)).columns)
