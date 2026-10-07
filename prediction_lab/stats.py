"""Statistical tests.

Two families:
1. Model comparison: is model M really better than the reference baseline, or is
   the gap noise? Paired per-round score differences, bootstrap CI, sign-flip
   permutation test, Holm correction across all models compared.
2. Process tests on the raw multipliers: do rounds look independent? Do class
   frequencies match the assumed crash distribution?
"""
import numpy as np
from scipy import stats as st

from .classes import EDGES, LABELS, N_CLASSES, theoretical_probs, to_class


# ------------------------------------------------------------ model comparison

def bootstrap_ci(d, n_boot=2000, alpha=0.05, block=1, seed=0):
    """CI for mean(d). block>1 uses a moving-block bootstrap (robust to autocorrelation)."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    if block <= 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        means = d[idx].mean(axis=1)
    else:
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
        idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
        means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def sign_flip_pvalue(d, n_perm=5000, seed=0):
    """One-sided p-value for H1: mean(d) < 0 (model's loss lower than reference).

    Under H0 (no difference) each paired difference is equally likely to have
    either sign, so we randomly flip signs and see how often the mean is as low.
    """
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    observed = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    return float((np.sum(null <= observed) + 1) / (n_perm + 1))


def holm(pvalues: dict) -> dict:
    """Holm–Bonferroni adjusted p-values (controls family-wise error)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def compare_to_reference(loss_model, loss_ref, seed=0):
    d = np.asarray(loss_model) - np.asarray(loss_ref)
    lo, hi = bootstrap_ci(d, seed=seed)
    return {
        "mean_diff": float(d.mean()),   # negative = model better
        "ci95": (lo, hi),
        "p_one_sided": sign_flip_pvalue(d, seed=seed),
    }


# ------------------------------------------------------------ process tests

def ljung_box(x, lags=20):
    """Ljung–Box test for autocorrelation up to `lags`. Small p => dependence."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = (x ** 2).sum()
    acf = np.array([(x[k:] * x[:-k]).sum() / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return {"acf": acf.tolist(), "Q": float(q), "p": float(st.chi2.sf(q, lags)),
            "acf_band95": float(1.96 / np.sqrt(n))}


def runs_test(x):
    """Wald–Wolfowitz runs test above/below the median. Small p => non-random ordering."""
    x = np.asarray(x, float)
    med = np.median(x)
    s = x[x != med] > med
    n1, n2 = s.sum(), (~s).sum()
    if n1 == 0 or n2 == 0:
        return {"runs": None, "z": None, "p": None}
    runs = 1 + np.sum(s[1:] != s[:-1])
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var)
    return {"runs": int(runs), "expected": float(mu), "z": float(z), "p": float(2 * st.norm.sf(abs(z)))}


def transition_test(x, segment):
    """Chi-square test: is the current class independent of the previous class?"""
    x = np.asarray(x, float)
    segment = np.asarray(segment)
    c = to_class(x)
    same = segment[1:] == segment[:-1]
    T = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(T, (c[:-1][same], c[1:][same]), 1)
    T = T[T.sum(axis=1) > 0][:, T.sum(axis=0) > 0]
    if T.shape[0] < 2 or T.shape[1] < 2:
        return {"chi2": None, "p": None}
    chi2, p, dof, expected = st.chi2_contingency(T)
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p),
            "min_expected": float(expected.min())}


def distribution_vs_theory(x, rtp=0.97):
    """Goodness of fit of class counts to the assumed crash distribution."""
    c = np.bincount(to_class(x), minlength=N_CLASSES)
    exp = theoretical_probs(rtp) * c.sum()
    chi2, p = st.chisquare(c, exp)
    return {"labels": LABELS, "observed": c.tolist(), "expected": exp.tolist(),
            "chi2": float(chi2), "p": float(p)}


def implied_rtp(x, thresholds=(1.5, 2, 3, 5, 10, 20)):
    """Under P(X>=m)=RTP/m, m*P(X>=m) estimates RTP at every threshold m."""
    x = np.asarray(x, float)
    out = {}
    for m in thresholds:
        k = int((x >= m).sum())
        p = k / len(x)
        se = np.sqrt(p * (1 - p) / len(x))
        out[str(m)] = {"estimate": float(m * p), "se": float(m * se), "count": k}
    return out
