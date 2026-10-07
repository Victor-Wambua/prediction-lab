import numpy as np

from prediction_lab.classes import N_CLASSES, to_class
from prediction_lab.evaluation import walk_forward
from prediction_lab.metrics import per_row_brier, per_row_log_loss, summarize
from prediction_lab.models import GlobalFrequency, Model, RecentWindow, Uniform
from prediction_lab.simulation import generate_random_rounds


class Spy(Model):
    """Records the size of every training set so we can prove no future rows leak in."""
    name = "spy"

    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        self.n_train = len(y)

    def predict_proba(self, X_row, y_hist):
        self.calls.append((X_row.index[0], self.n_train, len(y_hist)))
        return np.full(N_CLASSES, 0.2)


def test_walk_forward_never_trains_on_target_or_future():
    x = generate_random_rounds(300, seed=0)
    spy = Spy()
    res = walk_forward(x, models=[GlobalFrequency(), spy], initial_train=100, holdout=0.0)
    # Only the walk-forward calls (the later in-sample overfit pass is deliberate).
    calls = spy.calls[: len(res.eval_index)]
    assert [c[0] for c in calls] == list(res.eval_index)
    for t, n_train, n_hist in calls:
        assert n_train == t and n_hist == t  # trained on exactly rounds 0..t-1


def test_rolling_window_limits_history():
    x = generate_random_rounds(300, seed=0)
    spy = Spy()
    res = walk_forward(x, models=[GlobalFrequency(), spy], initial_train=100, window=50, holdout=0.0)
    assert all(n == 50 for _, n, _ in spy.calls[: len(res.eval_index)])


def test_holdout_excluded_by_default():
    x = generate_random_rounds(500, seed=0)
    res = walk_forward(x, models=[GlobalFrequency(), Uniform()], initial_train=100, holdout=0.2)
    assert res.eval_index.max() < 400
    fin = walk_forward(x, models=[GlobalFrequency(), Uniform()], initial_train=100, holdout=0.2, final=True)
    assert fin.eval_index.min() >= 400


class Cheater(Model):
    """Deliberately leaky: it is handed the true labels. Must be flagged as 'too good'."""
    name = "cheater"

    def __init__(self, y_true):
        self.y_true = y_true

    def predict_proba(self, X_row, y_hist):
        p = np.full(N_CLASSES, 0.05)
        p[self.y_true[X_row.index[0]]] = 0.8
        return p


def test_leaky_model_is_detected_as_significant():
    """Positive control for the significance machinery itself."""
    x = generate_random_rounds(600, seed=5)
    res = walk_forward(x, models=[GlobalFrequency(), RecentWindow(50), Cheater(to_class(x))],
                       initial_train=100, holdout=0.0)
    sig = res.significance()
    assert sig["cheater"]["significant"]
    assert not sig["recent_50"]["significant"]


def test_metrics_basics():
    y = np.array([0, 1])
    P = np.array([[1.0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]])
    assert per_row_brier(y, P).tolist() == [0.0, 0.5]
    assert np.isclose(per_row_log_loss(y, P)[1], np.log(2))
    s = summarize(y, P)
    assert s["accuracy"] == 0.5  # argmax ties resolve to class 0
