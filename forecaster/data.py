"""Data schema and loaders.

Internal format is one row per game ("wide"):
    gameid, date, league, blue_team, red_team, blue_win, length_min, lead15_k (optional),
    b_top..b_sup, r_top..r_sup

`from_oracles_elixir` converts the public Oracle's Elixir match-data CSV (one row per player)
into that format. The data itself is not included in this repository; download it from
https://oracleselixir.com/tools/downloads and respect its terms of use.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .constants import POSITIONS

BLUE_COLS = [f"b_{p}" for p in POSITIONS]
RED_COLS = [f"r_{p}" for p in POSITIONS]
CHAMP_COLS = BLUE_COLS + RED_COLS
GAME_COLUMNS = ["gameid", "date", "league", "blue_team", "red_team", "blue_win",
                "length_min", "lead15_k"] + CHAMP_COLS


def validate_games(games: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy or raise ValueError describing what is wrong."""
    g = games.copy()
    if "lead15_k" not in g.columns:
        g["lead15_k"] = np.nan
    missing = [c for c in GAME_COLUMNS if c not in g.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    g["date"] = pd.to_datetime(g["date"])
    if g["gameid"].duplicated().any():
        raise ValueError("duplicate gameid values")
    if g[CHAMP_COLS + ["length_min", "blue_win"]].isna().any().any():
        raise ValueError("NaN in champions / length / result")
    if not g["blue_win"].isin([0, 1]).all():
        raise ValueError("blue_win must be 0 or 1")
    if not g["length_min"].between(10, 90).all():
        raise ValueError("length_min outside the plausible 10-90 minute range")
    return g.sort_values("date").reset_index(drop=True)


def from_oracles_elixir(source, leagues=None) -> pd.DataFrame:
    """Convert an Oracle's Elixir CSV (path or DataFrame) into the wide game table."""
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_csv(source, low_memory=False)
    else:
        df = source.copy()
    need = {"gameid", "date", "league", "side", "position", "champion",
            "teamname", "result", "gamelength"}
    if not need <= set(df.columns):
        raise ValueError(f"CSV is missing columns: {sorted(need - set(df.columns))}")
    df = df[df["position"].isin(POSITIONS)].copy()
    if leagues:
        df = df[df["league"].isin(leagues)]
    df["side"] = df["side"].str.lower()
    counts = df.groupby("gameid").size()
    df = df[df["gameid"].isin(counts[counts == 10].index)]

    champs = df.pivot_table(index="gameid", columns=["side", "position"],
                            values="champion", aggfunc="first")
    wide = pd.DataFrame(index=champs.index)
    for p in POSITIONS:
        wide[f"b_{p}"] = champs[("blue", p)]
        wide[f"r_{p}"] = champs[("red", p)]
    meta = df.groupby("gameid").agg(date=("date", "first"), league=("league", "first"),
                                    length_min=("gamelength", lambda s: s.iloc[0] / 60.0))
    blue = df[df["side"] == "blue"].groupby("gameid").agg(
        blue_team=("teamname", "first"), blue_win=("result", "first"))
    red = df[df["side"] == "red"].groupby("gameid").agg(red_team=("teamname", "first"))
    out = wide.join(meta).join(blue).join(red).dropna().reset_index()
    out["blue_win"] = out["blue_win"].astype(int)
    out["lead15_k"] = np.nan
    return validate_games(out[GAME_COLUMNS])


def champion_index(games: pd.DataFrame) -> dict[str, list[str]]:
    """Champions seen per position (used by the UI)."""
    idx = {}
    for p in POSITIONS:
        names = pd.concat([games[f"b_{p}"], games[f"r_{p}"]]).unique().tolist()
        idx[p] = sorted(names)
    return idx
