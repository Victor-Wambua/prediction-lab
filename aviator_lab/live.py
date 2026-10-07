"""Live prediction loop, run by the collector after each new round.

1. Resolve pending predictions whose target round has now arrived: the target is
   the round with seq = base.seq + 1 in the SAME segment. If the segment changed
   (a gap), pending predictions are voided rather than scored against an
   unknown round.
2. Fit the live models on everything stored so far and predict the NEXT round.
   predicted_at is written before the next round can be observed; the report
   checks that this ordering holds for every scored prediction.
"""
import numpy as np

from . import database as db, targets, tiers
from .classes import LABELS, to_class
from .features import build_features, features_for_next
from .models import STREAK_FEATURES, GlobalFrequency, Logistic, Markov1, RecentWindow, Theoretical

PROB_COLS = db.PROB_COLS


class LivePredictor:
    def __init__(self, retrain_every=25):
        self.models = [GlobalFrequency(), RecentWindow(50), Markov1(), Theoretical(0.97),
                       Logistic(STREAK_FEATURES, name="logreg_streaks", retrain_every=retrain_every),
                       Logistic(None, name="logreg_all", retrain_every=retrain_every)]
        self.last_fit = {}

    def resolve(self, conn):
        pending = conn.execute(
            "SELECT p.*, r.segment_id AS seg, r.seq AS seq FROM predictions p"
            " JOIN rounds r ON r.id = p.base_round_pk WHERE p.status='pending' AND p.run_id=0"
        ).fetchall()
        n = 0
        for p in pending:
            target = conn.execute("SELECT * FROM rounds WHERE segment_id=? AND seq=?",
                                  (p["seg"], p["seq"] + 1)).fetchone()
            if target is None:
                continue
            probs = np.array([p[c] for c in PROB_COLS])
            db.resolve_prediction(conn, p["id"], target["id"], target["multiplier"],
                                  int(to_class(target["multiplier"])), probs, commit=False)
            n += 1
        tiers.resolve_pending(conn)
        targets.resolve_pending(conn)
        conn.commit()
        return n

    def predict_next(self, conn, segment_id):
        df = db.load_rounds(conn)
        x, seg = df["multiplier"].to_numpy(), df["segment_id"].to_numpy()
        y = to_class(x)
        X = build_features(x, seg)
        X_next = features_for_next(df.loc[df.segment_id == segment_id, "multiplier"].to_numpy())
        base_pk = int(df.loc[df.segment_id == segment_id, "id"].iloc[-1])
        out = {}
        for m in self.models:
            if len(x) < m.min_train:
                continue
            lf = self.last_fit.get(m.name)
            if lf is None or len(x) - lf >= m.retrain_every:
                m.fit(X, y)
                self.last_fit[m.name] = len(x)
            probs = m.predict_proba(X_next, y)
            db.insert_prediction(conn, m.name, base_pk, probs, commit=False)
            out[m.name] = probs
        seg_x = df.loc[df.segment_id == segment_id, "multiplier"].to_numpy()
        self.tier_preds = tiers.predict_all(x, seg_x)
        tiers.log_predictions(conn, base_pk, self.tier_preds)
        self.target_preds = targets.log_predictions(conn, base_pk, x)
        conn.commit()
        return out

    def on_new_rounds(self, conn, segment_id, new_pks):
        resolved = self.resolve(conn)
        preds = self.predict_next(conn, segment_id)
        if resolved:
            row = conn.execute(
                "SELECT COUNT(*) n, AVG(correct) acc, AVG(log_loss) ll FROM predictions"
                " WHERE run_id=0 AND status='resolved' AND model_name='global_freq'").fetchone()
            if row["n"]:
                print(f"  live global_freq: n={row['n']} acc={row['acc']:.1%} logloss={row['ll']:.3f}")
        name = "global_freq"  # the reference; logreg_all is noisy and overfits
        if name in preds:
            dist = "  ".join(f"{l}={p:.0%}" for l, p in zip(LABELS, preds[name]))
            print(f"  next ({name}): {dist}  [estimated probabilities, not certainty]")
        print(f"  next round reaches at least {self.target_preds[0.5]:.2f}x  (50% level)")
        tp = self.tier_preds["global_freq"]
        print("  next tiers (global_freq): " + "  ".join(
            f"{l}={p:.0%}" for l, p in zip(tiers.TIER_SHORT, tp)))
