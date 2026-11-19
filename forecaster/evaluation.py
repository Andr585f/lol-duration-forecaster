"""Honest evaluation: walk-forward splits, bootstrap CIs, coverage and a negative control.

Skill is measured *beyond a league-average baseline* (`base`), so league differences alone
cannot masquerade as draft signal. `r_skill` is corr(pred - base, y - base).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import interval_coverage
from .data import CHAMP_COLS, validate_games
from .model import DurationModel


def walk_forward(games: pd.DataFrame, n_splits: int = 5, initial_frac: float = 0.4,
                 model_factory=DurationModel) -> pd.DataFrame:
    """Expanding-window evaluation: always train on the past, test on the next block."""
    g = validate_games(games)
    n = len(g)
    edges = np.linspace(int(n * initial_frac), n, n_splits + 1).astype(int)
    parts = []
    for i in range(n_splits):
        train, test = g.iloc[:edges[i]], g.iloc[edges[i]:edges[i + 1]]
        model = model_factory().fit(train)
        out = model.predict_frame(test)
        lg_mean = train.groupby("league")["length_min"].mean()
        out["base"] = test["league"].map(lg_mean).fillna(train["length_min"].mean()).values
        out["length_min"] = test["length_min"].values
        out["lead15_k"] = test["lead15_k"].values
        out["fold"] = i
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def _stats(res: pd.DataFrame) -> dict:
    y, p, b = res["length_min"].values, res["pred_min"].values, res["base"].values
    dp, dy = p - b, y - b
    r = float(np.corrcoef(dp, dy)[0, 1]) if dp.std() > 0 else 0.0
    return {"r_skill": r,
            "mae": float(np.mean(np.abs(y - p))),
            "mae_baseline": float(np.mean(np.abs(y - b))),
            "mae_gain": float(np.mean(np.abs(y - b)) - np.mean(np.abs(y - p)))}


def bootstrap_ci(res: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    point = _stats(res)
    draws = {k: [] for k in ("r_skill", "mae_gain")}
    for _ in range(n_boot):
        s = _stats(res.iloc[rng.integers(0, len(res), len(res))])
        for k in draws:
            draws[k].append(s[k])
    out = dict(point)
    for k, v in draws.items():
        out[f"{k}_ci95"] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return out


def shuffle_champions(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Negative control: break the draft-to-length link by permuting champion columns jointly."""
    g = validate_games(games)
    perm = np.random.default_rng(seed).permutation(len(g))
    shuffled = g.copy()
    shuffled[CHAMP_COLS] = g[CHAMP_COLS].values[perm]
    return shuffled


def full_report(games: pd.DataFrame, n_boot: int = 1000) -> dict:
    res = walk_forward(games)
    real = bootstrap_ci(res, n_boot)
    real["coverage_80"] = interval_coverage(res["length_min"], res["pred_min"], res["sigma"], 0.8)
    real["n_test_games"] = int(len(res))
    ctrl_res = walk_forward(shuffle_champions(games))
    ctrl = bootstrap_ci(ctrl_res, n_boot)
    ctrl["note"] = "champions shuffled; the CI for r_skill should contain 0"
    return {"model": real, "negative_control": ctrl}
