"""Synthetic crash data with KNOWN properties, used to test the pipeline itself.

- Null data: i.i.d. rounds from the standard crash distribution. Nothing is
  predictable beyond the base rates, so no model may significantly beat the
  global-frequency baseline. If one does, the pipeline is leaking.
- Planted-signal data: same, except after a run of `streak` rounds below 1.5x the
  next round is boosted with probability `strength`. A sound pipeline should
  detect this (a positive control: it proves the tests have power to find a
  signal when one really exists).

Measured power (seed 7, walk-forward, Holm 5%): a boost after ANY sub-1.5x round
(strength 0.3) is detected at 3k rounds; a boost only after 3 lows in a row is
NOT detected at 5k. Patterns rarer or weaker than that can hide at these sample
sizes. Absence of evidence at small n is not evidence of absence.
"""
import numpy as np

from .evaluation import walk_forward


def _draw(rng, n, rtp):
    u = rng.random(n)
    x = np.floor(100 * rtp / u) / 100  # P(X >= m) ≈ rtp / m, floored to cents
    return np.maximum(x, 1.00)


def generate_random_rounds(n=10_000, rtp=0.97, seed=42):
    return _draw(np.random.default_rng(seed), n, rtp)


def generate_signal_rounds(n=10_000, rtp=0.97, strength=0.3, streak=1, boost=3.0, seed=42):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    run = 0
    for i in range(n):
        v = _draw(rng, 1, rtp)[0]
        if run >= streak and rng.random() < strength:
            v = max(v, np.floor(100 * boost * rtp / rng.random()) / 100)
        x[i] = v
        run = run + 1 if v < 1.5 else 0
    return x


def run_sanity_checks(n=5000, seed=42, retrain_every=50, progress=False):
    """Run the full walk-forward pipeline on null and planted-signal data."""
    out = {}
    for kind, x in (("null", generate_random_rounds(n, seed=seed)),
                    ("signal", generate_signal_rounds(n, seed=seed))):
        res = walk_forward(x, retrain_every=retrain_every, holdout=0.0, progress=progress)
        sig = res.significance()
        winners = [k for k, v in sig.items() if v.get("significant")]
        # Theoretical model uses the true generating RTP here, so it may "win"
        # legitimately on synthetic data; only learned models count.
        learned_winners = [w for w in winners if w.startswith(("logreg", "markov", "recent"))]
        out[kind] = {"result": res, "learned_winners": learned_winners}
    out["null_pass"] = not out["null"]["learned_winners"]
    out["signal_pass"] = bool(out["signal"]["learned_winners"])
    return out
