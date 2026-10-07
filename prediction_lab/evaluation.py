"""Walk-forward evaluation.

For every evaluated round t, each model is (re)fit only on rounds [lo, t) and then
predicts round t. `lo` is 0 (expanding window) or t - window (rolling window).
All models are scored on exactly the same rounds so their scores are comparable.

A final holdout (the most recent `holdout` fraction) is excluded by default.
Only look at it with final=True, and rarely: every look at it makes it less
"untouched".
"""
from dataclasses import dataclass, field

import numpy as np

from . import metrics, stats
from .classes import N_CLASSES, to_class
from .features import build_features
from .models import REFERENCE_MODEL, default_models


@dataclass
class WalkForwardResult:
    eval_index: np.ndarray            # row indices that were predicted
    y: np.ndarray                     # actual classes of those rows
    probs: dict                       # model name -> (n_eval, 5) array
    models: list
    config: dict
    train_scores: dict = field(default_factory=dict)  # in-sample log loss, overfit check

    def leaderboard(self):
        rows = []
        for m in self.models:
            s = metrics.summarize(self.y, self.probs[m.name], m.probabilistic)
            s["model"] = m.name
            s["train_log_loss"] = self.train_scores.get(m.name)
            rows.append(s)
        return rows

    def significance(self, reference=REFERENCE_MODEL):
        """Each probabilistic model vs reference on per-round log loss (+ accuracy)."""
        ref_ll = metrics.per_row_log_loss(self.y, self.probs[reference])
        ref_hit = (self.probs[reference].argmax(1) == self.y).astype(float)
        out, pvals = {}, {}
        for m in self.models:
            if m.name == reference or not m.probabilistic:
                continue
            ll = metrics.per_row_log_loss(self.y, self.probs[m.name])
            hit = (self.probs[m.name].argmax(1) == self.y).astype(float)
            r = stats.compare_to_reference(ll, ref_ll)
            acc = stats.compare_to_reference(-hit, -ref_hit)  # negative = more hits
            r["accuracy_diff"] = -acc["mean_diff"]
            r["accuracy_p_one_sided"] = acc["p_one_sided"]
            out[m.name] = r
            pvals[m.name] = r["p_one_sided"]
        for name, p_adj in stats.holm(pvals).items():
            out[name]["p_holm"] = p_adj
            out[name]["significant"] = bool(p_adj < 0.05)
        return out

    def to_results(self):
        return {"leaderboard": self.leaderboard(), "significance": self.significance()}


def walk_forward(x, segment=None, models=None, initial_train=100, window=None,
                 holdout=0.2, final=False, retrain_every=25, progress=False):
    x = np.asarray(x, float)
    n = len(x)
    segment = np.zeros(n, int) if segment is None else np.asarray(segment)
    models = models or default_models(retrain_every=retrain_every)
    X = build_features(x, segment)
    y = to_class(x)

    dev_end = int(round(n * (1 - holdout)))
    start = max(initial_train, max(m.min_train for m in models))
    if final:
        start, end = max(start, dev_end), n
    else:
        end = dev_end
    if end - start < 20:
        raise ValueError(f"Not enough rounds to evaluate (have {n}, need start={start} < end={end}).")

    eval_index = np.arange(start, end)
    probs = {m.name: np.zeros((len(eval_index), N_CLASSES)) for m in models}
    last_fit = {m.name: None for m in models}

    for i, t in enumerate(eval_index):
        lo = 0 if window is None else max(0, t - window)
        X_row = X.iloc[[t]]
        y_hist = y[lo:t]
        for m in models:
            lf = last_fit[m.name]
            if lf is None or t - lf >= m.retrain_every:
                m.fit(X.iloc[lo:t], y_hist)
                last_fit[m.name] = t
            probs[m.name][i] = m.predict_proba(X_row, y_hist)
        if progress and i % 500 == 0:
            print(f"  walk-forward {i}/{len(eval_index)}", flush=True)

    # Overfitting check: fit once on the training span preceding the final
    # prediction point and score in-sample. A big gap vs test log loss = overfit.
    train_scores = {}
    lo = 0 if window is None else max(0, end - 1 - window)
    Xtr, ytr = X.iloc[lo:end - 1], y[lo:end - 1]
    for m in models:
        if not m.probabilistic:
            continue
        m.fit(Xtr, ytr)
        P = np.vstack([m.predict_proba(Xtr.iloc[[j]], ytr[:j]) for j in range(len(ytr))])
        train_scores[m.name] = float(metrics.per_row_log_loss(ytr, P).mean())

    cfg = {"initial_train": initial_train, "window": window, "holdout": holdout,
           "final": final, "retrain_every": retrain_every, "start": int(start), "end": int(end)}
    return WalkForwardResult(eval_index, y[eval_index], probs, models, cfg, train_scores)


def format_leaderboard(result: WalkForwardResult) -> str:
    sig = result.significance()
    lines = [f"{'MODEL':<20}{'ACC':>8}{'BAL_ACC':>9}{'LOGLOSS':>9}{'TRAIN_LL':>10}{'BRIER':>8}{'ECE':>7}"
             f"{'ΔLL vs ref':>12}{'p(holm)':>9}"]
    for r in sorted(result.leaderboard(), key=lambda r: (r["log_loss"] is None, r["log_loss"] or 0)):
        s = sig.get(r["model"], {})
        fmt = lambda v, f: (f.format(v) if v is not None else "—")
        lines.append(
            f"{r['model']:<20}{r['accuracy']:>8.1%}{r['balanced_accuracy']:>9.1%}"
            f"{fmt(r['log_loss'], '{:.4f}'):>9}{fmt(r['train_log_loss'], '{:.4f}'):>10}"
            f"{fmt(r['brier'], '{:.4f}'):>8}{fmt(r['ece'], '{:.3f}'):>7}"
            f"{fmt(s.get('mean_diff'), '{:+.4f}'):>12}{fmt(s.get('p_holm'), '{:.3f}'):>9}"
            + ("  *" if s.get("significant") else "")
        )
    lines.append(f"(reference = {REFERENCE_MODEL}; ΔLL<0 means lower log loss than reference; "
                 f"* = significant after Holm correction at 5%)")
    return "\n".join(lines)
