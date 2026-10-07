"""Prediction classes. Edges are [lo, hi) in multiplier units."""
import numpy as np

EDGES = np.array([1.5, 3.0, 5.0, 10.0])
LABELS = ["1.00–1.49x", "1.50–2.99x", "3.00–4.99x", "5.00–9.99x", "10x+"]
N_CLASSES = len(LABELS)


def to_class(x):
    """Multiplier(s) -> class index 0..4. Works on scalars and arrays."""
    return np.searchsorted(EDGES, np.asarray(x, dtype=float), side="right")


def theoretical_probs(rtp: float = 0.97) -> np.ndarray:
    """Class probabilities if P(X >= m) = rtp / m for m > 1 (standard crash model).

    This is an ASSUMPTION about the game, not a measured fact. It is used as one
    baseline and to generate synthetic data; the empirical baselines do not use it.
    """
    cuts = np.concatenate(([1.0], EDGES, [np.inf]))
    surv = np.where(np.isinf(cuts), 0.0, rtp / cuts)
    surv[0] = 1.0  # everything is >= 1.00
    return surv[:-1] - surv[1:]
