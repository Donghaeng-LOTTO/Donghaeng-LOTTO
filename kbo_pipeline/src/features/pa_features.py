"""plate_appearances 파생 컬럼.

- is_top / batting_team_code / fielding_team_code
- is_lotte_batting / is_lotte_fielding
- lotte_score_before / opponent_score_before / score_diff_lotte_before
- scoring_position_before / late_clutch
- final_win_label_lotte / home_win / away_win
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _is_top_half(home_or_away) -> bool | None:
    """relay의 homeOrAway: '0' = 원정 공격(초), '1' = 홈 공격(말)인 케이스가 일반적."""
    s = str(home_or_away)
    if s in ("0", "away", "AWAY", "top"):
        return True
    if s in ("1", "home", "HOME", "bot", "bottom"):
        return False
    return None


def add_pa_derived_columns(
    pa_df: pd.DataFrame,
    games_detail: pd.DataFrame,
    lotte_code: str = config.LOTTE_TEAM_CODE,
) -> pd.DataFrame:
    if pa_df.empty:
        return pa_df

    pa = pa_df.copy()
    pa["game_id"] = pa["game_id"].astype(str)

    gd = games_detail.copy()
    gd["game_id"] = gd["game_id"].astype(str)
    gd_small = gd[[
        "game_id", "away_team_code", "home_team_code",
        "away_score", "home_score", "winner_team_code",
    ]]

    pa = pa.merge(gd_small, on="game_id", how="left")

    pa["is_top"] = pa["home_or_away"].apply(_is_top_half)

    pa["batting_team_code"] = np.where(
        pa["is_top"].eq(True), pa["away_team_code"], pa["home_team_code"])
    pa["fielding_team_code"] = np.where(
        pa["is_top"].eq(True), pa["home_team_code"], pa["away_team_code"])

    pa["is_lotte_batting"] = pa["batting_team_code"].eq(lotte_code)
    pa["is_lotte_fielding"] = pa["fielding_team_code"].eq(lotte_code)
    is_lotte_game = pa["is_lotte_batting"] | pa["is_lotte_fielding"]

    # 롯데 기준 점수 (사전 상태 컬럼이 있을 때만)
    if {"away_score_before", "home_score_before"}.issubset(pa.columns):
        lotte_is_home = pa["home_team_code"].eq(lotte_code)
        pa["lotte_score_before"] = np.where(
            lotte_is_home, pa["home_score_before"], pa["away_score_before"])
        pa["opponent_score_before"] = np.where(
            lotte_is_home, pa["away_score_before"], pa["home_score_before"])
        pa.loc[~is_lotte_game, ["lotte_score_before", "opponent_score_before"]] = np.nan
        pa["score_diff_lotte_before"] = (
            pa["lotte_score_before"] - pa["opponent_score_before"])
    else:
        pa["lotte_score_before"] = np.nan
        pa["opponent_score_before"] = np.nan
        pa["score_diff_lotte_before"] = np.nan

    # 득점권
    if {"base2_before", "base3_before"}.issubset(pa.columns):
        pa["scoring_position_before"] = (
            pa["base2_before"].fillna(0).astype(int)
            + pa["base3_before"].fillna(0).astype(int)
        ) > 0
    else:
        pa["scoring_position_before"] = None

    # 후반 접전: 7회 이후 & 점수차 2점 이내
    if {"away_score_before", "home_score_before"}.issubset(pa.columns):
        diff = (pa["home_score_before"] - pa["away_score_before"]).abs()
        pa["late_clutch"] = (
            pd.to_numeric(pa["inning"], errors="coerce") >= 7) & (diff <= 2)
    else:
        pa["late_clutch"] = None

    # 최종 승패 라벨 (winner 미상이면 NaN 유지 위해 float로)
    pa["home_win"] = pa["winner_team_code"].eq(pa["home_team_code"]).astype(float)
    pa["away_win"] = pa["winner_team_code"].eq(pa["away_team_code"]).astype(float)
    pa.loc[pa["winner_team_code"].isna(), ["home_win", "away_win"]] = np.nan

    pa["final_win_label_lotte"] = np.where(
        ~is_lotte_game, np.nan,
        np.where(pa["winner_team_code"].eq(lotte_code), 1.0,
                 np.where(pa["winner_team_code"].eq("DRAW"), 0.5, 0.0)),
    )

    return pa
