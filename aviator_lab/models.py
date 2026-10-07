"""Models. All of them output a probability vector over the 5 classes.

Interface:
    fit(X_train, y_train)            -- rows strictly before the prediction point
    predict_proba(X_row, y_hist)     -- X_row: 1-row DataFrame, y_hist: labels before t
Attributes:
    retrain_every -- refit cadence in rounds during walk-forward
    min_train     -- rounds required before the model makes predictions
    probabilistic -- False for hard-guess models (proba metrics are skipped)
"""
import numpy as np

from .classes import N_CLASSES, theoretical_probs

PRIOR_STRENGTH = 5.0  # pseudo-counts pulling sparse estimates toward the global mix


def _freq(y, alpha=1.0):
    c = np.bincount(np.asarray(y, dtype=int), minlength=N_CLASSES).astype(float) + alpha
    return c / c.sum()


class Model:
    name = "model"
    probabilistic = True
    retrain_every = 1
    min_train = 1

    def fit(self, X, y):
        pass

    def predict_proba(self, X_row, y_hist) -> np.ndarray:
        raise NotImplementedError


class Uniform(Model):
    name = "uniform"

    def predict_proba(self, X_row, y_hist):
        return np.full(N_CLASSES, 1.0 / N_CLASSES)


class Theoretical(Model):
    """Baseline from an ASSUMED RTP, not from data. Useful as a reference point."""

    def __init__(self, rtp=0.97):
        self.rtp = rtp
        self.name = f"theoretical_rtp{int(round(rtp * 100))}"

    def predict_proba(self, X_row, y_hist):
        return theoretical_probs(self.rtp)


class GlobalFrequency(Model):
    """Baseline A: frequency of each class over all prior rounds."""
    name = "global_freq"

    def fit(self, X, y):
        self.p = _freq(y)

    def predict_proba(self, X_row, y_hist):
        return self.p


class RecentWindow(Model):
    """Baseline B: frequency over the last `window` rounds, shrunk toward global."""

    def __init__(self, window):
        self.window = window
        self.name = f"recent_{window}"

    def fit(self, X, y):
        self.prior = _freq(y)

    def predict_proba(self, X_row, y_hist):
        recent = np.asarray(y_hist[-self.window:], dtype=int)
        c = np.bincount(recent, minlength=N_CLASSES) + PRIOR_STRENGTH * self.prior
        return c / c.sum()


class RandomGuess(Model):
    """Draws a class at random with the observed class frequencies (hard guess)."""
    name = "random_guess"
    probabilistic = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.p = _freq(y)

    def predict_proba(self, X_row, y_hist):
        out = np.zeros(N_CLASSES)
        out[self.rng.choice(N_CLASSES, p=self.p)] = 1.0
        return out


class Markov1(Model):
    """Baseline D (transitions): P(class_t | class_{t-1}) with shrinkage to global."""
    name = "markov_1"

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.prior = _freq(y)
        prev = _prev_class(X)
        T = np.zeros((N_CLASSES, N_CLASSES))
        ok = ~np.isnan(prev)
        np.add.at(T, (prev[ok].astype(int), y[ok]), 1.0)
        self.T = T + PRIOR_STRENGTH * self.prior
        self.T /= self.T.sum(axis=1, keepdims=True)

    def predict_proba(self, X_row, y_hist):
        prev = _prev_class(X_row)[0]
        return self.prior if np.isnan(prev) else self.T[int(prev)]


def _prev_class(X):
    cols = [f"prev1_is_c{c}" for c in range(N_CLASSES)]
    M = X[cols].to_numpy()
    out = np.full(len(M), np.nan)
    has = ~np.isnan(M).any(axis=1)
    out[has] = M[has].argmax(axis=1)
    return out


STREAK_FEATURES = ["streak_under_1.5", "streak_under_2", "streak_over_3", "streak_over_5",
                   "since_3.0x", "since_5.0x", "since_10.0x"]


class Logistic(Model):
    """Multinomial logistic regression on engineered features (regularised)."""

    def __init__(self, features=None, name="logreg_all", C=0.1, retrain_every=25, min_train=200):
        self.features = features
        self.name = name
        self.C = C
        self.retrain_every = retrain_every
        self.min_train = min_train

    def _cols(self, X):
        return X if self.features is None else X[self.features]

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # Imputer/scaler are fit on the training rows only -> no future normalisation.
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000),
        )
        self.pipe.fit(self._cols(X), y)
        self.classes_ = self.pipe.classes_

    def predict_proba(self, X_row, y_hist):
        p = self.pipe.predict_proba(self._cols(X_row))[0]
        out = np.full(N_CLASSES, 1e-6)
        out[self.classes_] = p
        return out / out.sum()


def default_models(retrain_every=25, seed=0):
    return [
        Uniform(),
        Theoretical(0.97),
        GlobalFrequency(),
        RecentWindow(10), RecentWindow(25), RecentWindow(50), RecentWindow(100),
        RandomGuess(seed),
        Markov1(),
        Logistic(STREAK_FEATURES, name="logreg_streaks", retrain_every=retrain_every),
        Logistic(None, name="logreg_all", retrain_every=retrain_every),
    ]


REFERENCE_MODEL = "global_freq"  # the baseline every other model must beat
