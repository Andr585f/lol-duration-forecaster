"""In-game nowcast: adjust the pre-game forecast using the gold lead at 15:00.

    live = pregame - slope * (min(|lead_k|, clamp) - ref_lead)

`lead_k` is the leading team's gold lead in thousands. The pre-game forecast already
averages over a typical lead, so the adjustment is centred on `ref_lead` (the mean clamped
lead in the fitting data); without centring the nowcast would be biased short. The slope is fitted from out-of-fold
residuals, with a bootstrap interval, rather than trusted blindly. Linearity beyond the
clamp is untested, so leads above it are not extrapolated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import NOWCAST_LEAD_CLAMP_K, NOWCAST_REF_LEAD_K, NOWCAST_SLOPE


def apply_nowcast(pre_pred, lead_k, slope: float = NOWCAST_SLOPE,
                  clamp: float = NOWCAST_LEAD_CLAMP_K, ref_lead: float = NOWCAST_REF_LEAD_K):
    lead = np.minimum(np.abs(np.asarray(lead_k, float)), clamp)
    return np.asarray(pre_pred, float) - slope * (lead - ref_lead)


def fit_slope(resid, lead_k, clamp: float = NOWCAST_LEAD_CLAMP_K,
              n_boot: int = 1000, seed: int = 0) -> dict:
    """Minutes saved per 1k of lead, with a 95% bootstrap interval."""
    x = np.minimum(np.abs(np.asarray(lead_k, float)), clamp)
    y = np.asarray(resid, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    est = -np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(-np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"slope": float(est), "ci95": [float(lo), float(hi)], "n": int(len(x)),
            "ref_lead": float(x.mean())}


def evaluate(res: pd.DataFrame, slope: float | None = None) -> dict:
    """Compare pre-game vs live forecasts on out-of-fold predictions that have a lead value."""
    d = res.dropna(subset=["lead15_k"])
    if len(d) < 30:
        return {"n": int(len(d)), "note": "too few games with a 15:00 gold lead"}
    s = fit_slope(d["length_min"] - d["pred_min"], d["lead15_k"]) if slope is None else None
    use = slope if slope is not None else s["slope"]
    ref = s["ref_lead"] if s else float(np.minimum(np.abs(d["lead15_k"]), NOWCAST_LEAD_CLAMP_K).mean())
    live = apply_nowcast(d["pred_min"], d["lead15_k"], use, ref_lead=ref)
    y = d["length_min"].values
    out = {
        "n": int(len(d)), "slope_used": float(use), "ref_lead_used": ref,
        "r_pregame": float(np.corrcoef(d["pred_min"], y)[0, 1]),
        "r_live": float(np.corrcoef(live, y)[0, 1]),
        "mae_pregame": float(np.mean(np.abs(y - d["pred_min"]))),
        "mae_live": float(np.mean(np.abs(y - live))),
    }
    if s:
        out["slope_fit"] = s
    return out
