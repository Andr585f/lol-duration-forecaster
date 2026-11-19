"""Synthetic match generator so the project runs without any external data.

The generating process is deliberately realistic in *difficulty*: the draft carries a small
signal buried in large noise (typical r around 0.1-0.2), so the evaluation numbers are
modest and honest rather than flattering. Champion names are fictional on purpose, so
nothing here can be mistaken for real balance findings.

Ground truth used to generate lengths (minutes):
    length = league_base + 4.5 * mean(effect * stage_weight) - 0.8 * gold_lead_at_15 + noise
where `stage_weight` is 1.0 for the winning side's champions and 0.7 for the losing side's.
That winner/loser asymmetry is exactly the hindsight trap the model must avoid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import POSITIONS, RANDOM_SEED
from .data import GAME_COLUMNS, validate_games

LEAGUE_BASE = {"North": 31.8, "South": 33.0, "East": 32.4, "West": 31.2}
TRUE_MULTIPLIER = 4.5
TRUE_LEAD_SLOPE = 0.8
NOISE_SD = 4.6

_SYLL = ["va", "ro", "ki", "sel", "mor", "tha", "lu", "zen", "bri", "kor", "ae", "dun",
         "fi", "gal", "hex", "ny", "ox", "pra", "qui", "ser", "tor", "ul", "wy", "yar"]


def _names(rng: np.random.Generator, n: int) -> list[str]:
    names: set[str] = set()
    while len(names) < n:
        k = int(rng.integers(2, 4))
        names.add("".join(rng.choice(_SYLL, size=k)).capitalize())
    return sorted(names)


def generate_games(n_games: int = 6000, seed: int = RANDOM_SEED,
                   start: str = "2024-01-01", champs_per_pos: int = 26) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = {p: _names(np.random.default_rng(seed + i), champs_per_pos)
            for i, p in enumerate(POSITIONS)}
    effect = {p: rng.normal(0, 0.55, champs_per_pos) for p in POSITIONS}
    pop = {}
    for p in POSITIONS:
        w = 1.0 / np.arange(1, champs_per_pos + 1) ** 0.7
        rng.shuffle(w)
        pop[p] = w / w.sum()

    leagues = list(LEAGUE_BASE)
    teams = {lg: [f"{lg[:2].upper()}-{i + 1}" for i in range(8)] for lg in leagues}
    strength = {t: rng.normal(0, 1) for lg in leagues for t in teams[lg]}

    dates = pd.Timestamp(start) + pd.to_timedelta(np.arange(n_games) // 4, unit="D")
    rows = []
    for gi in range(n_games):
        lg = leagues[int(rng.integers(len(leagues)))]
        bt, rt = rng.choice(teams[lg], size=2, replace=False)
        gap = strength[bt] - strength[rt]
        blue_win = int(rng.random() < 1 / (1 + np.exp(-(0.45 * gap + 0.1))))
        picks = {"b": [], "r": []}
        devs = {"b": [], "r": []}
        for side in ("b", "r"):
            won = blue_win if side == "b" else 1 - blue_win
            for p in POSITIONS:
                ci = int(rng.choice(champs_per_pos, p=pop[p]))
                picks[side].append(pool[p][ci])
                devs[side].append(effect[p][ci] * (1.0 if won else 0.7))
        lead = abs(rng.normal(0.55 * abs(gap), 2.0))
        length = (LEAGUE_BASE[lg]
                  + TRUE_MULTIPLIER * np.mean(devs["b"] + devs["r"])
                  - TRUE_LEAD_SLOPE * min(lead, 6.0)
                  + 1.4                       # offsets the mean lead effect
                  + rng.normal(0, NOISE_SD))
        row = {"gameid": f"SYN-{gi:06d}", "date": dates[gi], "league": lg,
               "blue_team": bt, "red_team": rt, "blue_win": blue_win,
               "length_min": float(np.clip(length, 18, 55)), "lead15_k": float(lead)}
        for j, p in enumerate(POSITIONS):
            row[f"b_{p}"] = picks["b"][j]
            row[f"r_{p}"] = picks["r"][j]
        rows.append(row)
    return validate_games(pd.DataFrame(rows)[GAME_COLUMNS])
