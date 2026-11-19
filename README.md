# LoL Duration Forecaster

A small forecasting project: predict how long a League of Legends pro match will last from the
draft, then update that forecast live from the gold lead at 15:00. The focus is on doing the
statistics honestly: no data leakage, walk-forward validation, confidence intervals, and a
negative control that proves the pipeline cannot find signal in noise.

> **Scope note:** this repository is the forecasting core of a larger private project. The rest of it is tied to a use case I've chosen not to publish.

```
pip install -r requirements.txt
python -m pytest -q                 # 13 tests
python -m forecaster.cli demo       # walk-forward evaluation on synthetic data
python -m forecaster.cli serve      # local web app on http://127.0.0.1:5050
```

## How it works

**Champion length deviation.** For each (champion, position, result) the model keeps the average
game length, shrunk toward the league average: `dev = (mean - league_mean) * n / (n + 3)`.
Shrinkage stops a champion seen three times from dominating a forecast.

**Hindsight is the trap.** Deviations conditioned on the result look more predictive than they
are, because the result is unknown before the game. Every forecast is therefore the average of
two scenarios (blue wins / red wins), each using the matching win-side or loss-side deviations.

**No in-sample leakage.** The scale (minutes per unit of deviation), league offsets, a graduated
prediction-bias correction and the forecast spread are all fitted on *out-of-fold* features, so a
game's feature never saw that game. A test asserts a forecast is unchanged when the game's own
length and result are altered.

**Live nowcast.** `live = pregame - slope * (min(|lead|, 6k) - typical_lead)`, where `lead` is the
leading team's gold lead at 15:00 in thousands. The slope is estimated with a bootstrap interval
rather than assumed. Leads above 6k are clamped because linearity there is untested.

## Validation (synthetic demo data, 3,600 walk-forward test games)

| Check | Result |
|---|---|
| Skill beyond a league-average baseline, `r` | 0.074, 95% CI [0.041, 0.107] |
| MAE improvement over baseline | 0.009 min, CI [0.001, 0.016] |
| 80% interval coverage | 79.4% |
| **Negative control** (champions shuffled), `r` | -0.015, CI [-0.047, 0.018] contains 0 |
| Nowcast slope recovered (true value 0.8) | 0.76 min per 1k, CI [0.63, 0.87] |
| Correlation with outcome, pre-game vs live | 0.14 -> 0.24 |

Reading these honestly: **the draft explains very little of game length.** The improvement over
a baseline is real but tiny in minutes, and the forecast spread is about 4.6 minutes. That is the
point of the validation tooling: it tells you how much to trust the model. On 3,000 games the
signal was not statistically detectable at all, which is why the test suite uses 8,000.

## Data

The repository ships **no real match data**. It runs on a synthetic generator with fictional
champions, built so the draft carries a small signal inside large noise. Synthetic results
demonstrate that the pipeline works and is unbiased; they say nothing about real-world accuracy.

To use real data, download the match-data CSVs from [Oracle's Elixir](https://oracleselixir.com/tools/downloads)
(please follow their terms of use and credit them), then:

```
python -m forecaster.cli evaluate --csv path/to/matches.csv --out reports/real.json
python -m forecaster.cli serve --csv path/to/matches.csv
```

The CSV adapter (`forecaster/data.py`) expects the standard player-row format. Real data has no
15:00 gold column, so the nowcast section is skipped unless you add `lead15_k`.

## Layout

```
forecaster/
  data.py         schema validation, Oracle's Elixir adapter
  synthetic.py    seeded synthetic match generator
  model.py        shrunk deviations, two-scenario features, out-of-fold fitting
  calibration.py  graduated bias bands, interval coverage, reliability curve
  evaluation.py   walk-forward splits, bootstrap CIs, negative control
  nowcast.py      gold-lead update with fitted, bootstrapped slope
  app.py          Flask app (localhost) + templates/index.html
  cli.py          demo / evaluate / serve
tests/            13 tests incl. leakage, symmetry, negative control
```

## Limitations

- Normal-distribution intervals; real game lengths are right-skewed.
- The nowcast slope in the demo is fitted on the same evaluation block it is scored on (one
  parameter, so optimism is small, but a held-out refit would be stricter).
- Linearity of the gold-lead effect is assumed inside the 0-6k range and not tested beyond it.
- Not designed or validated for any use beyond studying match-length forecasting.
