"""Calibration helpers: graduated bias correction, interval coverage, reliability curve."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .constants import BIAS_BINS, BIAS_SHRINK


def fit_bias_bands(pred: np.ndarray, resid: np.ndarray, n_bins: int = BIAS_BINS,
                   shrink: float = BIAS_SHRINK):
    """Mean residual per prediction-quantile bin, shrunk toward zero.

    Returns (centers, corrections) for use with `np.interp`. Replaces a single flat bias
    constant: forecasts that are systematically too short in one range and too long in
    another get a different correction in each range.
    """
    pred = np.asarray(pred, float)
    resid = np.asarray(resid, float)
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([pred.mean()]), np.array([0.0])
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    centers, corr = [], []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() == 0:
            continue
        centers.append(pred[m].mean())
        corr.append(resid[m].mean() * m.sum() / (m.sum() + shrink))
    return np.array(centers), np.array(corr)


def apply_bias_bands(pred: np.ndarray, centers: np.ndarray, corr: np.ndarray) -> np.ndarray:
    if len(centers) < 2:
        return np.asarray(pred, float) + (corr[0] if len(corr) else 0.0)
    return np.asarray(pred, float) + np.interp(pred, centers, corr)


def interval_coverage(y, pred, sigma, level: float = 0.8) -> float:
    """Share of outcomes inside the central `level` interval. Should be close to `level`."""
    z = norm.ppf(0.5 + level / 2)
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean(np.abs(y - pred) <= z * np.asarray(sigma)))


def reliability_curve(y, pred, sigma, cutoff_min: float, n_bins: int = 5):
    """Reliability of P(game length <= cutoff): predicted probability vs observed frequency."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    p = norm.cdf((cutoff_min - pred) / np.asarray(sigma))
    obs = (y <= cutoff_min).astype(float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    out = []
    for b in range(max(len(edges) - 1, 1)):
        m = which == b
        if m.sum():
            out.append({"predicted": float(p[m].mean()), "observed": float(obs[m].mean()),
                        "n": int(m.sum())})
    return out
