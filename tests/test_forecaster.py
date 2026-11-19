import numpy as np
import pandas as pd
import pytest

from forecaster import evaluation, nowcast
from forecaster.app import create_app
from forecaster.calibration import fit_bias_bands, interval_coverage
from forecaster.data import CHAMP_COLS, from_oracles_elixir, validate_games
from forecaster.model import DurationModel
from forecaster.synthetic import generate_games


@pytest.fixture(scope="module")
def games():
    return generate_games(n_games=8000, seed=3)


def test_synthetic_is_valid_and_deterministic(games):
    again = generate_games(n_games=8000, seed=3)
    pd.testing.assert_frame_equal(games, again)
    validate_games(games)


def test_validate_rejects_bad_data(games):
    bad = games.copy()
    bad.loc[0, "length_min"] = 3
    with pytest.raises(ValueError):
        validate_games(bad)


def test_prediction_ignores_own_outcome(games):
    """A game's forecast must not depend on that game's length or result (no leakage)."""
    train, test = games.iloc[:6000], games.iloc[6000:6100].copy()
    m = DurationModel().fit(train)
    p1 = m.predict_frame(test)["pred_min"].values
    test["length_min"] = 99.0
    test["blue_win"] = 1 - test["blue_win"]
    p2 = m.predict_frame(test)["pred_min"].values
    np.testing.assert_allclose(p1, p2)


def test_team_swap_symmetry(games):
    m = DurationModel().fit(games)
    row = games.iloc[[0]].copy()
    swapped = row.copy()
    for p in ["top", "jng", "mid", "bot", "sup"]:
        swapped[f"b_{p}"], swapped[f"r_{p}"] = row[f"r_{p}"].values, row[f"b_{p}"].values
    a = m.predict_frame(row)["pred_min"].iloc[0]
    b = m.predict_frame(swapped)["pred_min"].iloc[0]
    assert a == pytest.approx(b)


def test_walk_forward_only_uses_past(games):
    res = evaluation.walk_forward(games, n_splits=4)
    assert len(res) == len(games) - int(len(games) * 0.4)
    assert res["fold"].is_monotonic_increasing


def test_model_beats_baseline_on_signal(games):
    res = evaluation.walk_forward(games)
    ci = evaluation.bootstrap_ci(res, n_boot=300)
    assert ci["r_skill"] > 0
    assert ci["r_skill_ci95"][0] > 0, "real draft signal should be detected"


def test_negative_control_finds_nothing(games):
    ctrl = evaluation.walk_forward(evaluation.shuffle_champions(games))
    ci = evaluation.bootstrap_ci(ctrl, n_boot=300)
    assert ci["r_skill_ci95"][0] <= 0 <= ci["r_skill_ci95"][1]


def test_interval_coverage_is_reasonable(games):
    res = evaluation.walk_forward(games)
    cov = interval_coverage(res["length_min"], res["pred_min"], res["sigma"], 0.8)
    assert 0.72 < cov < 0.88


def test_bias_bands_flat_when_no_bias():
    rng = np.random.default_rng(0)
    pred = rng.normal(32, 1, 2000)
    centers, corr = fit_bias_bands(pred, rng.normal(0, 4, 2000))
    assert np.abs(corr).max() < 0.7


def test_nowcast_recovers_known_slope(games):
    res = evaluation.walk_forward(games)
    d = res.dropna(subset=["lead15_k"])
    fit = nowcast.fit_slope(d["length_min"] - d["pred_min"], d["lead15_k"], n_boot=300)
    assert fit["ci95"][0] < 0.8 < fit["ci95"][1]
    ev = nowcast.evaluate(res)
    assert ev["mae_live"] < ev["mae_pregame"]


def test_nowcast_clamps_extreme_leads():
    a = nowcast.apply_nowcast(32.0, 6.0)
    b = nowcast.apply_nowcast(32.0, 15.0)
    assert a == pytest.approx(b)


def test_oracles_elixir_adapter_roundtrip(games):
    rows = []
    for _, g in games.head(20).iterrows():
        for side, pre in (("Blue", "b"), ("Red", "r")):
            for p in ["top", "jng", "mid", "bot", "sup"]:
                rows.append({"gameid": g.gameid, "date": g.date, "league": g.league,
                             "side": side, "position": p, "champion": g[f"{pre}_{p}"],
                             "teamname": g.blue_team if side == "Blue" else g.red_team,
                             "result": g.blue_win if side == "Blue" else 1 - g.blue_win,
                             "gamelength": g.length_min * 60})
    out = from_oracles_elixir(pd.DataFrame(rows))
    assert len(out) == 20
    src = games.set_index("gameid").loc[out["gameid"]].reset_index()
    assert (out["blue_win"].values == src["blue_win"].values).all()
    assert (out[CHAMP_COLS].values == src[CHAMP_COLS].values).all()


def test_app_endpoints(games):
    app = create_app(games=games)
    c = app.test_client()
    meta = c.get("/api/meta").get_json()
    blue = {p: meta["champions"][p][0] for p in meta["positions"]}
    red = {p: meta["champions"][p][1] for p in meta["positions"]}
    r = c.post("/api/forecast", json={"blue": blue, "red": red, "lead15_k": 3, "cutoff_min": 30})
    d = r.get_json()
    assert r.status_code == 200
    assert d["live_pred_min"] < d["pred_min"]
    assert 0 <= d["p_ends_by_cutoff"] <= 1
    assert c.post("/api/forecast", json={"blue": "x", "red": {}}).status_code == 400
    assert c.post("/api/forecast", json={"blue": {}, "red": {}, "lead15_k": "abc"}).status_code == 400
    assert c.get("/").status_code == 200
