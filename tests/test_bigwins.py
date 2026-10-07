import numpy as np

from aviator_lab import bigwins
from aviator_lab.simulation import generate_random_rounds


def test_rounds_since_and_gaps_respect_segments():
    ev = np.array([1, 0, 0, 1, 0, 1, 0, 0], bool)
    seg = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    since = bigwins.rounds_since(ev, seg)
    assert np.isnan(since[0]) and since[1:4].tolist() == [1, 2, 3]
    assert np.isnan(since[4]) and np.isnan(since[5]) and since[6:].tolist() == [1, 2]
    assert bigwins.gaps(ev, seg).tolist() == [3]  # 3->5 crosses a segment: not a gap


def test_null_data_shows_no_pattern():
    x = generate_random_rounds(20_000, seed=3)
    res, adj = bigwins.analyse(x, np.zeros(len(x), int))
    r10 = res[10.0]
    assert r10.enough and abs(r10.rate - 0.097) < 0.01
    assert r10.freq_p > 0.01
    assert all(p > 0.05 for p in adj.values())


def test_planted_due_effect_is_detected():
    """If big rounds really became likelier after long droughts, the test must see it."""
    rng = np.random.default_rng(0)
    x = generate_random_rounds(20_000, seed=4)
    since = 0
    for i in range(len(x)):
        if since > 15 and rng.random() < 0.15:
            x[i] = max(x[i], 12.0)
        since = 0 if x[i] >= 10 else since + 1
    res, adj = bigwins.analyse(x, np.zeros(len(x), int))
    assert adj[(10.0, "due")] < 0.01


def test_current_drought():
    x = np.array([1.2, 150.0, 1.1, 1.3, 2.0])
    r = bigwins.analyse_threshold(x, np.zeros(5, int), m=100.0)
    assert r.current_drought == 3 and not r.drought_is_lower_bound
    assert np.isclose(r.drought_chance, (1 - 0.0097) ** 3)
