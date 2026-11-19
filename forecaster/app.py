"""Local Flask app: pick ten champions, get a duration forecast; optionally add a 15:00 gold lead."""
from __future__ import annotations

import numpy as np
from flask import Flask, jsonify, render_template, request

from .constants import NOWCAST_LEAD_CLAMP_K, NOWCAST_REF_LEAD_K, POSITIONS
from .data import champion_index, from_oracles_elixir
from .model import DurationModel
from .nowcast import apply_nowcast
from .synthetic import generate_games


def create_app(csv: str | None = None, games=None) -> Flask:
    app = Flask(__name__)
    if games is None:
        games = from_oracles_elixir(csv) if csv else generate_games()
    model = DurationModel().fit(games)
    champs = champion_index(games)
    leagues = sorted(games["league"].unique().tolist())
    ref = float(np.minimum(games["lead15_k"].abs(), NOWCAST_LEAD_CLAMP_K).mean()) if games["lead15_k"].notna().any() else NOWCAST_REF_LEAD_K
    app.config.update(REF_LEAD=ref, MODEL=model, SYNTHETIC=csv is None and games["gameid"].iloc[0].startswith("SYN"))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify({"positions": POSITIONS, "champions": champs, "leagues": leagues,
                        "n_train": model.n_train, "synthetic": app.config["SYNTHETIC"],
                        "sigma": model.sigma})

    @app.post("/api/forecast")
    def forecast():
        body = request.get_json(silent=True) or {}
        blue, red = body.get("blue", {}), body.get("red", {})
        if not isinstance(blue, dict) or not isinstance(red, dict):
            return jsonify({"error": "blue and red must be objects keyed by position"}), 400
        out = model.predict_one(blue, red, body.get("league"))
        lead = body.get("lead15_k")
        if lead is not None:
            try:
                lead = float(lead)
            except (TypeError, ValueError):
                return jsonify({"error": "lead15_k must be a number"}), 400
            out["live_pred_min"] = float(apply_nowcast(out["pred_min"], lead, ref_lead=app.config["REF_LEAD"]))
        cutoff = body.get("cutoff_min")
        if cutoff is not None:
            try:
                out["p_ends_by_cutoff"] = float(model.prob_ends_by(out["pred_min"], float(cutoff)))
            except (TypeError, ValueError):
                return jsonify({"error": "cutoff_min must be a number"}), 400
        return jsonify(out)

    return app
