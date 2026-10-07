import numpy as np

from prediction_lab import stats
from prediction_lab.classes import N_CLASSES, theoretical_probs, to_class
from prediction_lab.simulation import generate_random_rounds, generate_signal_rounds, run_sanity_checks


def test_synthetic_distribution_matches_theory():
    x = generate_random_rounds(200_000, seed=0)
    freq = np.bincount(to_class(x), minlength=N_CLASSES) / len(x)
    assert np.allclose(freq, theoretical_probs(0.97), atol=0.006)
    assert x.min() >= 1.0


def test_independence_tests_quiet_on_null_data():
    x = generate_random_rounds(5000, seed=1)
    assert stats.ljung_box(np.log(x))["p"] > 0.01
    assert stats.runs_test(x)["p"] > 0.01
    assert stats.transition_test(x, np.zeros(len(x)))["p"] > 0.01


def test_independence_tests_fire_on_planted_signal():
    x = generate_signal_rounds(5000, seed=1)
    assert stats.transition_test(x, np.zeros(len(x)))["p"] < 0.01


def test_pipeline_null_and_signal_controls():
    """The whole pipeline: no false discovery on i.i.d. data, detection on planted signal."""
    out = run_sanity_checks(n=3000, seed=7, retrain_every=100)
    assert out["null_pass"], out["null"]["learned_winners"]
    assert out["signal_pass"]


def test_holm():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
