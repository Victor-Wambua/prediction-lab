# Aviator Research Lab

A research experiment that asks one question:

> After observing 10,000+ Aviator rounds, does any model predict the next round's
> class better than a properly constructed baseline, on data it has never seen?

"No" is a successful result. The lab is built so that it can prove us wrong in
either direction, and so that it is hard to fool ourselves.

**Observe only.** The collector reads the round-history text that the game already
shows you. It never clicks, types, bets, cashes out, or touches the betting
controls. You log in yourself in the opened browser, and no credentials are stored.

> ## ⚠️ Disclaimer
>
> - **Do not use this on any website or service whose terms of use, policies or
>   applicable law restrict or forbid automated access, scraping, monitoring or
>   data collection.** Many betting sites do. It is your responsibility to read and
>   follow the terms of any site before running the collector against it. If they
>   restrict this kind of use, don't run it there.
> - This is an educational statistics and machine-learning experiment. It is
>   **not** a betting tool, a betting system or financial advice, and it makes no
>   claim that crash-game outcomes can be predicted. The expected result is that
>   they can't.
> - Nothing on the dashboard (predictions, tiers, "at least X" targets, big-multiplier
>   statistics) should be used to decide whether, when or how much to bet.
>   Gambling involves the risk of losing money. If gambling is causing you harm,
>   please seek help from a local support service.
> - Not affiliated with, endorsed by or connected to Betika, Spribe or any game
>   operator. All trademarks belong to their owners.
> - **No real data is included.** This repository contains no collected rounds,
>   predictions, results, databases or account information. Every multiplier in the
>   code, tests and documentation is dummy data: hand-written test values or numbers
>   generated at random by the built-in simulator. Any resemblance to real results
>   from any betting website or game is purely coincidental. Data you collect
>   yourself stays in your local `data/` folder, which is excluded from git.
> - Provided "as is", without warranty of any kind. You use it at your own risk
>   and are solely responsible for how you use it.

## Setup (macOS)

```bash
brew install python@3.12
git clone https://github.com/Victor-Wambua/prediction-lab.git
cd prediction-lab
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env        # optional settings, no secrets
pytest -q                   # ~40 tests, ~20s
```

## Workflow

| Step | Command | What it does |
|---|---|---|
| 0 | `python -m aviator_lab simulate` | Runs the whole pipeline on synthetic data. Must PASS before trusting real results. |
| 1 | `python -m aviator_lab probe` | Opens the site. Log in, open the game, press Enter. It prints where it found the history strip and a suggested selector. |
| 2 | `python -m aviator_lab collect` (or `./run.sh`) | Records rounds and logs a live prediction for every next round. Ctrl-C to stop. |
| 3 | `python -m aviator_lab report` | Data-quality checks, distribution vs the RTP model, independence tests, live scores. |
| 4 | `python -m aviator_lab evaluate` | Walk-forward backtest of all models, with significance tests. Results are saved to the DB. |
| 5 | `python -m aviator_lab dashboard` | Streamlit dashboard at http://localhost:8501. |

Useful options:

```bash
python -m aviator_lab collect --selector "<selector printed by probe>"  # pin the strip selector
python -m aviator_lab collect --order oldest_first        # if probe shows the oldest round first
python -m aviator_lab evaluate --window 500               # rolling window instead of expanding
python -m aviator_lab evaluate --retrain-every 100
python -m aviator_lab evaluate --data synthetic-null --n 10000 --seed 3
python -m aviator_lab evaluate --final                    # final holdout: rarely, and never tune afterwards
python -m aviator_lab verify-round --server-seed S --client-seed A --client-seed B --client-seed C --multiplier 2.41 --hash H
```

Run **one** collector at a time. A second one refuses to start, because two
collectors on one database would record every round twice.

## Dashboard sections

| Section | What it shows |
|---|---|
| **Next round prediction** | "At least X.XXx" at a chosen chance level (50/30/20/10%). This is the highest multiplier that share of past rounds reached. The last 40 outcomes are shown green (reached) or red (crashed below), with % correct over time against what chance alone gives. |
| **Risk tiers** | Next-round probabilities for Low (≤2.00x), Middle (2.01–5.00x) and High (>5.00x), plus a track record. |
| **Research classes** | Next-round probabilities for the 5 classes from every model, plus a live scoreboard and rolling log loss. |
| **Big multipliers** | Counts and rates for 10x/50x/100x/1000x+ against fair-game rates, current droughts, and tests for "is it due", clustering and time of day (corrected for multiple tests). |
| **Distribution / backtest / data quality** | Class mix against the fair-game model, walk-forward leaderboards with significance, and collector health checks. |

Every prediction on the dashboard is logged **before** its round and scored
after it. If past rounds carry no information, each "at least X" level's hit rate
converges to the level itself, and no model beats `global_freq`.

## How it avoids fooling itself

**Collection**
- Rounds are identified by **sequence alignment**, not by value (1.37x happens many
  times). Each new strip snapshot must match the stored tail on at least 8 consecutive
  rounds. If continuity can't be proven, the collector opens a new *segment*
  instead of guessing. This survives refreshes and restarts because the tail is read
  from the DB.
- A snapshot is processed only after two identical reads (to skip mid-animation
  reads). If any item fails to parse, the whole snapshot is rejected. There is no
  "read the whole page" fallback.
- Rounds from the initial strip are flagged `is_bootstrap` because their real times are unknown.

**Evaluation**
- Feature row *t* depends only on rounds before *t* in the same segment. A test
  corrupts round *t* and every later round and checks that row *t* doesn't change.
- Walk-forward evaluation trains on `[0, t)` and predicts `t`, for every *t*. A spy
  model test checks this, and every model is scored on the same rounds.
- Imputation and scaling are fit on training rows only.
- The most recent 20% of rounds is a held-out final test set that `evaluate` excludes by default.
- **Reference baseline = `global_freq`** (empirical class frequencies so far). Every
  model is compared with it on per-round log loss using a paired sign-flip
  permutation test, a bootstrap CI and a Holm correction across models.
- `TRAIN_LL` vs `LOGLOSS` shows overfitting. For example, `logreg_all` on null data
  scores train 1.44 vs test 1.54.
- Live predictions are timestamped before their target round is observed, and the
  report checks this. A prediction is only scored against the next round in the
  same segment, and is voided across gaps.

**Controls**
- `simulate` runs everything on i.i.d. synthetic data (no learned model may win)
  and on data with a **planted** pattern (the pipeline must find it). A test also
  checks that a deliberately leaky "cheater" model is flagged.
- Measured power: a 30% boost after any sub-1.5x round is detected at about 3k rounds.
  A boost only after 3 lows in a row is **not** detected at 5k. Rare or weak
  patterns can hide at these sample sizes, so "not detected" at small *n* means
  little.

## Models

| Model | What it is |
|---|---|
| `global_freq` | **Reference.** Class frequencies over all prior rounds (baseline A). |
| `recent_10/25/50/100` | Frequencies over the last N rounds, shrunk toward global (baseline B). |
| `markov_1` | P(class \| previous class) (transitions). |
| `logreg_streaks` | Logistic regression on streak and recency features (baseline D, tested, not assumed). |
| `logreg_all` | Logistic regression on all features: rolling stats, distribution, streaks, lags (baseline C features). |
| `theoretical_rtp97` | P(X≥m)=0.97/m. An **assumption** about the game, shown for reference. |
| `uniform`, `random_guess` | Floors. `random_guess` draws a class from the observed frequencies. |

Classes: 1.00–1.49x, 1.50–2.99x, 3.00–4.99x, 5.00–9.99x, 10x+. Outputs are
probabilities: "48%" is the model's estimated chance, not confidence in the
everyday sense.

Accuracy is shown but is a weak metric. Under the RTP-97 model, always predicting
"1.00–1.49x" scores about 35%. Judge models on log loss, Brier and calibration
(ECE) relative to `global_freq`.

## Layout

```
aviator_lab/
  config.py       settings (.env)            classes.py    class edges + theoretical probs
  history.py      strip parsing + alignment   collector.py  Playwright observer, probe
  database.py     SQLite schema + access      live.py       live predict/resolve loop
  features.py     leakage-safe features       models.py     baselines + logistic
  evaluation.py   walk-forward + leaderboard  metrics.py    log loss, Brier, ECE, calibration
  stats.py        significance + independence tests
  simulation.py   null + planted-signal data  fairness.py   provably-fair hypotheses (separate)
  report.py       data quality + process report
  tiers.py        Low/Middle/High tier forecasts      targets.py   "at least X" predictions
  bigwins.py      big-multiplier frequency, drought, clustering, time-of-day tests
dashboard/app.py  Streamlit
tests/            pytest suite
```

DB tables: `segments`, `rounds`, `predictions` (live `run_id=0`, backtests
`run_id=model_runs.id`), `model_runs`, `tier_predictions` and `target_predictions`.

## Provably fair module

`fairness.py` only uses what the game's own "Provably fair" dialog shows for a
finished round. Spribe's exact derivation isn't assumed. Several candidate formulas
are tested, and `verify-round` reports which one, if any, reproduces the displayed
multiplier. Record several rounds and see which hypothesis survives all of them.
Verifying a round proves the result came from those seeds. It does not make the
next round predictable.

## Roadmap

- v0.4 (now): collector, schema, baselines, walk-forward, significance, simulation, dashboard, tiers, targets, big-multiplier analysis
- next: pin the real strip selector; collect 500 → 1,000 rounds; first `report`/`evaluate`
- then: Random Forest / XGBoost (they can catch threshold patterns that logistic misses), only once the pipeline is trusted
- then: record provably-fair data for ~20 rounds and run `verify-round`
- 10,000+ rounds: a single `--final` evaluation, then write up the result
