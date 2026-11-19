"""Leakage-free game-duration forecaster.

Idea: a champion's length deviation is the (shrunk) difference between its average game length
(by position and by whether its team won) and the league average.

Two traps this implementation avoids:

1. Hindsight inflation. Deviations are measured conditional on the result, but the result is
   unknown before the game. So every forecast is the average of two scenarios (blue wins /
   red wins), each using the matching win- or loss-side deviations.
2. In-sample leakage. The scale (minutes per unit of deviation), league offsets, bias bands
   and sigma are all fitted on *out-of-fold* features: a game's feature is computed from
   statistics that never saw that game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import apply_bias_bands, fit_bias_bands
from .constants import K_SHRINK, LEAGUE_SHRINK, POSITIONS, RANDOM_SEED, Z80
from .data import validate_games


def _long(g: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for p in POSITIONS:
        frames.append(pd.DataFrame({"champ": g[f"b_{p}"].values, "pos": p,
                                    "res": g["blue_win"].values, "len": g["length_min"].values}))
        frames.append(pd.DataFrame({"champ": g[f"r_{p}"].values, "pos": p,
                                    "res": 1 - g["blue_win"].values, "len": g["length_min"].values}))
    return pd.concat(frames, ignore_index=True)


class DurationModel:
    def __init__(self, k: float = K_SHRINK, n_folds: int = 5, seed: int = RANDOM_SEED,
                 fixed_multiplier: float | None = None):
        self.k, self.n_folds, self.seed = k, n_folds, seed
        self.fixed_multiplier = fixed_multiplier
        self.fitted = False

    # -- statistics -------------------------------------------------------------------------
    def _stats(self, g: pd.DataFrame):
        long = _long(g)
        ga = float(g["length_min"].mean())
        grp = long.groupby(["champ", "pos", "res"])["len"].agg(["mean", "count"])
        shrunk = (grp["mean"] - ga) * grp["count"] / (grp["count"] + self.k)
        return ga, shrunk.to_dict()

    @staticmethod
    def _features(g: pd.DataFrame, table: dict) -> np.ndarray:
        """Mean shrunk deviation over the 10 champions, averaged over both result scenarios."""
        a = np.zeros(len(g))
        b = np.zeros(len(g))
        for p in POSITIONS:
            bc, rc = g[f"b_{p}"].values, g[f"r_{p}"].values
            blue_w = np.array([table.get((c, p, 1), 0.0) for c in bc])
            blue_l = np.array([table.get((c, p, 0), 0.0) for c in bc])
            red_w = np.array([table.get((c, p, 1), 0.0) for c in rc])
            red_l = np.array([table.get((c, p, 0), 0.0) for c in rc])
            a += blue_w + red_l      # scenario A: blue wins
            b += blue_l + red_w      # scenario B: red wins
        return (a + b) / 20.0

    # -- fitting ----------------------------------------------------------------------------
    def fit(self, games: pd.DataFrame) -> "DurationModel":
        g = validate_games(games)
        n = len(g)
        if n < 50:
            raise ValueError("need at least 50 games to fit")
        folds = np.random.default_rng(self.seed).permutation(n) % self.n_folds
        feat = np.zeros(n)
        for f in range(self.n_folds):
            _, table = self._stats(g[folds != f])
            feat[folds == f] = self._features(g[folds == f], table)

        y = g["length_min"].values
        if self.fixed_multiplier is None:
            slope, intercept = np.polyfit(feat, y, 1)
            slope = max(float(slope), 0.0)           # a negative scale is noise, not signal
            intercept = float(np.mean(y - slope * feat))
        else:
            slope = float(self.fixed_multiplier)
            intercept = float(np.mean(y - slope * feat))
        self.multiplier, self.intercept = slope, intercept

        resid = y - (intercept + slope * feat)
        leagues = g["league"].values
        self.league_offset = {}
        for lg in np.unique(leagues):
            m = leagues == lg
            self.league_offset[lg] = float(resid[m].mean() * m.sum() / (m.sum() + LEAGUE_SHRINK))
        off = np.array([self.league_offset[lg] for lg in leagues])
        pred1 = intercept + slope * feat + off
        self.bias_centers, self.bias_corr = fit_bias_bands(pred1, y - pred1)
        pred2 = apply_bias_bands(pred1, self.bias_centers, self.bias_corr)
        self.sigma = float(np.std(y - pred2, ddof=1))

        self.league_mean, self.table = self._stats(g)
        self.n_train = n
        self.fitted = True
        return self

    # -- prediction -------------------------------------------------------------------------
    def predict_frame(self, games: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        feat = self._features(games, self.table)
        off = np.array([self.league_offset.get(lg, 0.0) for lg in games["league"].values])
        pred = apply_bias_bands(self.intercept + self.multiplier * feat + off,
                                self.bias_centers, self.bias_corr)
        return pd.DataFrame({"pred_min": pred, "sigma": self.sigma, "feature": feat,
                             "lo80": pred - Z80 * self.sigma, "hi80": pred + Z80 * self.sigma},
                            index=games.index)

    def predict_one(self, blue: dict, red: dict, league: str | None = None) -> dict:
        row = {"league": league or next(iter(self.league_offset), "?"), "blue_win": 0}
        for p in POSITIONS:
            row[f"b_{p}"] = blue.get(p, "")
            row[f"r_{p}"] = red.get(p, "")
        out = self.predict_frame(pd.DataFrame([row])).iloc[0]
        known = sum((c, p, 1) in self.table for p in POSITIONS for c in (blue.get(p), red.get(p)))
        return {"pred_min": float(out.pred_min), "sigma": float(out.sigma),
                "lo80": float(out.lo80), "hi80": float(out.hi80),
                "draft_feature": float(out.feature), "known_champion_slots": int(known)}

    def prob_ends_by(self, pred_min, minutes: float):
        """P(game length <= minutes) under the normal forecast distribution."""
        return norm.cdf((minutes - np.asarray(pred_min, float)) / self.sigma)
