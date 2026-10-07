"""Scoring. Probabilistic metrics are primary; accuracy is reported but secondary,
because always predicting the most common class already scores ~35%+."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from .classes import N_CLASSES

EPS = 1e-12


def per_row_log_loss(y, P):
    P = np.clip(np.asarray(P, float), EPS, 1.0)
    return -np.log(P[np.arange(len(y)), np.asarray(y)])


def per_row_brier(y, P):
    """Multiclass Brier: sum over classes of (p - onehot)^2. Range 0..2."""
    P = np.asarray(P, float)
    onehot = np.eye(N_CLASSES)[np.asarray(y)]
    return ((P - onehot) ** 2).sum(axis=1)


def summarize(y, P, probabilistic: bool = True) -> dict:
    y = np.asarray(y)
    P = np.asarray(P, float)
    pred = P.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, labels=range(N_CLASSES), average="macro", zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "log_loss": float(per_row_log_loss(y, P).mean()) if probabilistic else None,
        "brier": float(per_row_brier(y, P).mean()) if probabilistic else None,
        "ece": float(expected_calibration_error(y, P)) if probabilistic else None,
    }
    return out


def calibration_table(y, P, n_bins: int = 10):
    """One-vs-rest reliability over ALL (round, class) pairs.

    Each row: predicted probability bin -> mean predicted vs observed frequency.
    A calibrated model has mean_predicted ≈ observed in every populated bin.
    """
    import pandas as pd
    y = np.asarray(y)
    P = np.asarray(P, float)
    probs = P.ravel()
    hits = (np.eye(N_CLASSES)[y]).ravel()
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": probs, "hit": hits})
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("hit", "mean"), count=("p", "size"))
    return g.reset_index()


def expected_calibration_error(y, P, n_bins: int = 10) -> float:
    t = calibration_table(y, P, n_bins)
    w = t["count"] / t["count"].sum()
    return float((w * (t["mean_predicted"] - t["observed"]).abs()).sum())
