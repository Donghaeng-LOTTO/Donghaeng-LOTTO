"""경기 전(pre-game) 누적 기록 생성.

핵심 원칙 (데이터 누수 방지):
- boxscore를 선수별 날짜순으로 누적(cumsum)한 뒤 반드시 shift(1)을 적용한다.
- 따라서 각 행의 cum_* 값은 "해당 경기 직전까지"의 기록이다.
- 같은 경기의 최종 기록이 같은 경기의 예측 피처에 들어가는 일은 없다.

비율 스탯 한계 (boxscore 기반 MVP):
- OBP: HBP/SF가 boxscore에 없어 (H+BB)/(AB+BB) 근사 -> obp_approx
- SLG: 2루타/3루타가 boxscore에 없어 plate_appearances 기반 보강.
  PA 데이터가 없으면 비HR 안타를 전부 단타로 간주한 slg_lower_bound 사용.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _season_of(game_date: pd.Series) -> pd.Series:
    return game_date.astype(str).str.replace(r"\D", "", regex=True).str[:4].astype(int)


def _sort_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date_norm"] = (
        df["game_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    )
    df["season"] = df["game_date_norm"].str[:4].astype(int)
    return df.sort_values(["game_date_norm", "game_id"])


# ------------------------------------------------------------
# 타자
# ------------------------------------------------------------
def build_batter_pre_game_stats(
    boxscores: pd.DataFrame,
    pa_df: pd.DataFrame | None = None,
    cumulate_within_season: bool = True,
) -> pd.DataFrame:
    df = _sort_keys(boxscores)
    df["player_code"] = df["player_code"].astype(str)

    count_cols = ["ab", "hit", "hr", "bb", "kk", "rbi", "run"]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # PA 기반 2루타/3루타 보강 (있으면)
    if pa_df is not None and not pa_df.empty:
        pa = pa_df.copy()
        pa["batter_pcode"] = pa["batter_pcode"].astype(str)
        xbh = (
            pa.assign(
                is_2b=(pa["pa_result_type"] == "double").astype(int),
                is_3b=(pa["pa_result_type"] == "triple").astype(int),
            )
            .groupby(["game_id", "batter_pcode"])[["is_2b", "is_3b"]]
            .sum()
            .reset_index()
            .rename(columns={"batter_pcode": "player_code",
                             "is_2b": "doubles", "is_3b": "triples"})
        )
        xbh["game_id"] = xbh["game_id"].astype(str)
        xbh["player_code"] = xbh["player_code"].astype(str)
        xbh["xbh_source"] = "pa"
        df["game_id"] = df["game_id"].astype(str)
        df["player_code"] = df["player_code"].astype(str)
        df = df.merge(xbh, on=["game_id", "player_code"], how="left")
        df["xbh_source"] = df["xbh_source"].fillna("none")
        df["doubles"] = df["doubles"].fillna(0)
        df["triples"] = df["triples"].fillna(0)
        # df["xbh_source"] = np.where(
        #     df[["doubles", "triples"]].notna().all(axis=1), "pa", "none")
    else:
        df["doubles"] = 0.0
        df["triples"] = 0.0
        df["xbh_source"] = "none"

    group_keys = ["player_code", "season"] if cumulate_within_season else ["player_code"]
    grouped = df.groupby(group_keys, group_keys=False)

    cum_cols = count_cols + ["doubles", "triples"]
    for c in cum_cols:
        # cumsum 후 shift(1): 해당 경기 '이전까지' 누적
        df[f"cum_{c}"] = grouped[c].apply(lambda s: s.cumsum().shift(1)).fillna(0)
    df["games_before"] = grouped.cumcount()

    # 비율 스탯
    cum_ab = df["cum_ab"].replace(0, np.nan)
    cum_pa_denom = (df["cum_ab"] + df["cum_bb"]).replace(0, np.nan)

    df["avg_before"] = df["cum_hit"] / cum_ab
    df["obp_approx_before"] = (df["cum_hit"] + df["cum_bb"]) / cum_pa_denom

    singles = df["cum_hit"] - df["cum_doubles"] - df["cum_triples"] - df["cum_hr"]
    total_bases = (
        singles + 2 * df["cum_doubles"] + 3 * df["cum_triples"] + 4 * df["cum_hr"]
    )
    df["slg_before"] = total_bases / cum_ab
    df["ops_before"] = df["obp_approx_before"] + df["slg_before"]
    df["slg_is_lower_bound"] = (df["xbh_source"] == "none")

    out_cols = [
        "game_id", "game_date", "season", "team_side", "team_code",
        "player_code", "name", "games_before",
        "cum_ab", "cum_hit", "cum_hr", "cum_bb", "cum_kk", "cum_rbi", "cum_run",
        "cum_doubles", "cum_triples",
        "avg_before", "obp_approx_before", "slg_before", "ops_before",
        "slg_is_lower_bound",
    ]
    return df[out_cols].reset_index(drop=True)


# ------------------------------------------------------------
# 투수
# ------------------------------------------------------------
def build_pitcher_pre_game_stats(
    boxscores: pd.DataFrame,
    cumulate_within_season: bool = True,
) -> pd.DataFrame:
    df = _sort_keys(boxscores)
    df["pcode"] = df["pcode"].astype(str)

    count_cols = ["outs", "r", "er", "hit", "hr", "bb", "bbhp", "kk", "bf"]
    for c in count_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    group_keys = ["pcode", "season"] if cumulate_within_season else ["pcode"]
    grouped = df.groupby(group_keys, group_keys=False)

    for c in count_cols:
        df[f"cum_{c}"] = grouped[c].apply(lambda s: s.cumsum().shift(1)).fillna(0)
    df["games_before"] = grouped.cumcount()

    cum_outs = df["cum_outs"].replace(0, np.nan)
    cum_ip = cum_outs / 3.0

    df["ip_before"] = (df["cum_outs"] / 3.0)
    df["era_before"] = df["cum_er"] * 27.0 / cum_outs          # ER/IP*9
    df["whip_before"] = (df["cum_bb"] + df["cum_hit"]) / cum_ip
    df["k9_before"] = df["cum_kk"] * 27.0 / cum_outs
    df["bb9_before"] = df["cum_bb"] * 27.0 / cum_outs

    out_cols = [
        "game_id", "game_date", "season", "team_side", "team_code",
        "pcode", "name", "games_before",
        "cum_outs", "ip_before", "cum_r", "cum_er", "cum_hit", "cum_hr",
        "cum_bb", "cum_bbhp", "cum_kk", "cum_bf",
        "era_before", "whip_before", "k9_before", "bb9_before",
    ]
    return df[out_cols].reset_index(drop=True)


# ------------------------------------------------------------
# 실행 헬퍼
# ------------------------------------------------------------
def run_pre_game_stats(cumulate_within_season: bool = True) -> None:
    batters = pd.read_csv(
        config.PROCESSED_DIR / "batter_game_boxscores.csv",
        dtype={"game_id": str, "player_code": str},
    )
    pitchers = pd.read_csv(
        config.PROCESSED_DIR / "pitcher_game_boxscores.csv",
        dtype={"game_id": str, "pcode": str},
    )

    pa_path = config.PROCESSED_DIR / "plate_appearances.csv"
    pa_df = None
    if pa_path.exists():
        pa_df = pd.read_csv(pa_path, dtype={"game_id": str, "batter_pcode": str})

    batter_stats = build_batter_pre_game_stats(
        batters, pa_df=pa_df, cumulate_within_season=cumulate_within_season)
    pitcher_stats = build_pitcher_pre_game_stats(
        pitchers, cumulate_within_season=cumulate_within_season)

    b_out = config.PROCESSED_DIR / "batter_pre_game_stats.csv"
    p_out = config.PROCESSED_DIR / "pitcher_pre_game_stats.csv"
    batter_stats.to_csv(b_out, index=False, encoding="utf-8-sig")
    pitcher_stats.to_csv(p_out, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s rows=%d", b_out, len(batter_stats))
    logger.info("[saved] %s rows=%d", p_out, len(pitcher_stats))
