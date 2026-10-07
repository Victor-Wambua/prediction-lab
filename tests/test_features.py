import numpy as np
import pandas as pd

from prediction_lab.features import build_features, features_for_next
from prediction_lab.simulation import generate_random_rounds


def test_row_t_depends_only_on_past():
    """Core no-leakage property: row t is identical whether or not the future exists."""
    x = generate_random_rounds(400, seed=1)
    full = build_features(x)
    for t in [0, 1, 5, 49, 50, 51, 200, 399]:
        prefix = build_features(x[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], prefix.iloc[t], check_names=False)


def test_changing_current_and_future_values_does_not_change_row():
    x = generate_random_rounds(300, seed=2)
    t = 150
    y = x.copy()
    y[t:] = 999.0  # corrupt current + future
    pd.testing.assert_frame_equal(build_features(x).iloc[: t + 1], build_features(y).iloc[: t + 1])


def test_features_for_next_matches_build_features():
    x = generate_random_rounds(120, seed=3)
    nxt = features_for_next(x[:100])
    row = build_features(x).iloc[[100]].reset_index(drop=True)
    pd.testing.assert_frame_equal(nxt, row)


def test_features_do_not_cross_segments():
    x = generate_random_rounds(200, seed=4)
    seg = np.r_[np.zeros(100, int), np.ones(100, int)]
    f = build_features(x, seg)
    # First row of segment 2 has no history at all.
    assert f.iloc[100][["lag_1", "mean_5", "streak_under_1.5"]].isna().all()
    pd.testing.assert_frame_equal(f.iloc[100:].reset_index(drop=True), build_features(x[100:]))


def test_streak_and_recency_values():
    x = np.array([5.0, 1.1, 1.2, 1.3, 3.5, 1.0])
    f = build_features(x)
    assert list(f["streak_under_1.5"].iloc[1:]) == [0, 1, 2, 3, 0]
    assert list(f["since_3.0x"].iloc[1:]) == [0, 1, 2, 3, 0]
    assert f["lag_1"].iloc[4] == np.log(1.3)
