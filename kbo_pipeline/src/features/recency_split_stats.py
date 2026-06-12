"""Recency(이동평균) 및 좌우 스플릿 사전 기록 생성.

[Matchup2025] 근거:
  - 최근 성적(recency)이 인게임 의사결정에 중요.
  - 최근 500타석에 4배 가중하는 방식을 게임 단위로 단순화:
    rolling_5 (최근 5경기), rolling_10 (최근 10경기) 이동 평균.
  - shift(1) 적용으로 당해 경기 누수 방지.

[MC-MLB] / [Hirotsu2005] 근거:
  - 좌우 매치업(handedness)이 타격 확률에 유의미하게 영향.
  - vs_L / vs_R 스플릿 성적을 별도 피처로 제공.

출력:
  - processed/batter_recency_stats.csv
  - processed/pitcher_recency_stats.csv
  - processed/batter_split_stats.csv
  - processed/pitcher_split_stats.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

ID_DTYPES = {
    "game_id": str,
    "player_code": str,
    "pcode": str,
    "batter_pcode": str,
    "pitcher_pcode": str,
}


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def _sort_by_date(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_date_norm"] = (
        df["game_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    )
    df["season"] = df["_date_norm"].str[:4].astype(int)
    return df.sort_values(["_date_norm", "game_id"])


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """shift(1)이 포함된 rolling mean. 당해 경기 제외."""
    return (
        series.rolling(window=window, min_periods=1)
        .mean()
        .shift(1)
    )


# ---------------------------------------------------------------------------
# 타자 Recency
# ---------------------------------------------------------------------------
def build_batter_recency_stats(
    boxscores: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """타자 이동평균 기록 (최근 N경기 기준).

    Args:
        boxscores: batter_game_boxscores.csv
        windows: 이동 평균 창 크기 목록. 기본 [5, 10].

    Returns:
        game_id x player_code 단위 rolling 성적 DataFrame.
    """
    if windows is None:
        windows = [5, 10]

    df = _sort_by_date(boxscores, "player_code")
    df["player_code"] = df["player_code"].astype(str)

    num_cols = ["ab", "hit", "hr", "bb", "kk"]
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    # 단경기 비율 계산용 분모
    df["_ab_safe"] = df["ab"].replace(0, np.nan)
    df["_pa_safe"] = (df["ab"] + df["bb"]).replace(0, np.nan)

    out_frames = [df[["game_id", "game_date", "season", "team_code", "player_code"]].copy()]

    for w in windows:
        prefix = f"roll{w}"
        grp = df.groupby(["player_code", "season"], group_keys=False)

        roll_hit = grp["hit"].apply(lambda s: _rolling_mean(s, w))
        roll_ab  = grp["ab"].apply(lambda s: _rolling_mean(s, w))
        roll_bb  = grp["bb"].apply(lambda s: _rolling_mean(s, w))
        roll_hr  = grp["hr"].apply(lambda s: _rolling_mean(s, w))
        roll_kk  = grp["kk"].apply(lambda s: _rolling_mean(s, w))

        ab_safe = roll_ab.replace(0, np.nan)
        pa_safe = (roll_ab + roll_bb).replace(0, np.nan)

        df[f"{prefix}_avg"]  = roll_hit / ab_safe
        df[f"{prefix}_obp"]  = (roll_hit + roll_bb) / pa_safe
        df[f"{prefix}_hr_rate"] = roll_hr / ab_safe
        df[f"{prefix}_k_rate"]  = roll_kk / pa_safe
        df[f"{prefix}_bb_rate"] = roll_bb / pa_safe

        out_frames.append(
            df[[f"{prefix}_avg", f"{prefix}_obp",
                f"{prefix}_hr_rate", f"{prefix}_k_rate",
                f"{prefix}_bb_rate"]].copy()
        )

    result = pd.concat(out_frames, axis=1)
    logger.info("타자 recency 생성: %d행", len(result))
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 투수 Recency
# ---------------------------------------------------------------------------
def build_pitcher_recency_stats(
    boxscores: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """투수 이동평균 기록 (최근 N경기 기준).

    Args:
        boxscores: pitcher_game_boxscores.csv
        windows: 이동 평균 창 크기 목록. 기본 [5, 10].
    """
    if windows is None:
        windows = [5, 10]

    df = _sort_by_date(boxscores, "pcode")
    df["pcode"] = df["pcode"].astype(str)

    num_cols = ["outs", "er", "r", "hit", "bb", "kk", "hr", "bf"]
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    out_frames = [df[["game_id", "game_date", "season", "team_code", "pcode"]].copy()]

    for w in windows:
        prefix = f"roll{w}"
        grp = df.groupby(["pcode", "season"], group_keys=False)

        roll_outs = grp["outs"].apply(lambda s: _rolling_mean(s, w))
        roll_er   = grp["er"].apply(lambda s: _rolling_mean(s, w))
        roll_hit  = grp["hit"].apply(lambda s: _rolling_mean(s, w))
        roll_bb   = grp["bb"].apply(lambda s: _rolling_mean(s, w))
        roll_kk   = grp["kk"].apply(lambda s: _rolling_mean(s, w))
        roll_hr   = grp["hr"].apply(lambda s: _rolling_mean(s, w))
        roll_bf   = grp["bf"].apply(lambda s: _rolling_mean(s, w))

        ip_safe   = (roll_outs / 3.0).replace(0, np.nan)
        bf_safe   = roll_bf.replace(0, np.nan)

        df[f"{prefix}_era"]    = roll_er * 9.0 / ip_safe
        df[f"{prefix}_whip"]   = (roll_hit + roll_bb) / ip_safe
        df[f"{prefix}_k9"]     = roll_kk * 9.0 / ip_safe
        df[f"{prefix}_bb9"]    = roll_bb * 9.0 / ip_safe
        df[f"{prefix}_hr9"]    = roll_hr * 9.0 / ip_safe
        df[f"{prefix}_k_pct"]  = roll_kk / bf_safe
        df[f"{prefix}_bb_pct"] = roll_bb / bf_safe

        out_frames.append(
            df[[f"{prefix}_era", f"{prefix}_whip", f"{prefix}_k9",
                f"{prefix}_bb9", f"{prefix}_hr9",
                f"{prefix}_k_pct", f"{prefix}_bb_pct"]].copy()
        )

    result = pd.concat(out_frames, axis=1)
    logger.info("투수 recency 생성: %d행", len(result))
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 좌우 스플릿 — 타자
# ---------------------------------------------------------------------------
def build_batter_split_stats(
    pa_df: pd.DataFrame,
    player_map_df: pd.DataFrame,
    min_pa: int = 20,
) -> pd.DataFrame:
    """타자의 vs_L / vs_R 스플릿 사전 기록.

    Args:
        pa_df: plate_appearances.csv (pa_result_type 포함).
        player_map_df: player_id_map.csv (투수 throw_hand 포함).
        min_pa: 스플릿 기준 최소 타석 수. 미달 시 NaN.

    Returns:
        game_id x batter_pcode 단위 vs_L/vs_R 스플릿 통계.
    """
    df = pa_df.copy()
    df["batter_pcode"] = df["batter_pcode"].astype(str)
    df["pitcher_pcode"] = df.get("pitcher_pcode_norm", df.get("pitcher_pcode", "")).astype(str)

    # 투수 손잡이 조인
    pm = player_map_df.copy()
    pm["naver_pcode"] = pm["naver_pcode"].astype(str)
    hand_map = pm.set_index("naver_pcode")["throw_hand"].to_dict()
    df["_pitcher_hand"] = df["pitcher_pcode"].map(hand_map)

    def _normalize_hand(h) -> str | None:
        if pd.isna(h):
            return None
        s = str(h).strip().upper()
        if s in {"L", "좌", "좌투"}:
            return "L"
        if s in {"R", "우", "우투"}:
            return "R"
        return None

    df["_pitcher_hand"] = df["_pitcher_hand"].map(_normalize_hand)
    df = df[df["_pitcher_hand"].isin(["L", "R"])].copy()

    # 타석 결과를 hit / ab / bb / hr / kk 로 변환
    df["_is_ab"] = ~df["pa_result_type"].isin(["walk", "intentional_walk", "hit_by_pitch", "sac_bunt", "sac_fly"])
    df["_is_hit"] = df["pa_result_type"].isin(["single", "double", "triple", "home_run"])
    df["_is_hr"]  = df["pa_result_type"].eq("home_run")
    df["_is_bb"]  = df["pa_result_type"].isin(["walk", "intentional_walk", "hit_by_pitch"])
    df["_is_kk"]  = df["pa_result_type"].eq("strikeout")
    df["_is_2b"]  = df["pa_result_type"].eq("double")
    df["_is_3b"]  = df["pa_result_type"].eq("triple")

    # 날짜 기반 정렬 (as-of 조인용)
    if "game_date" in df.columns:
        df["_date_norm"] = df["game_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    else:
        df["_date_norm"] = df["game_id"].astype(str).str[:8]
    df["season"] = df["_date_norm"].str[:4].astype(int)
    df = df.sort_values(["_date_norm", "game_id"])

    records = []
    for hand in ("L", "R"):
        sub = df[df["_pitcher_hand"] == hand].copy()
        grp = sub.groupby(["batter_pcode", "season"], group_keys=False)

        for col, raw in [("ab", "_is_ab"), ("hit", "_is_hit"), ("hr", "_is_hr"),
                         ("bb", "_is_bb"), ("kk", "_is_kk")]:
            sub[f"cum_{col}"] = grp[raw].apply(
                lambda s: s.astype(int).cumsum().shift(1)
            ).fillna(0)

        cum_ab = sub["cum_ab"].replace(0, np.nan)
        cum_pa = (sub["cum_ab"] + sub["cum_bb"]).replace(0, np.nan)

        sub[f"vs{hand}_avg"]     = sub["cum_hit"] / cum_ab
        sub[f"vs{hand}_obp"]     = (sub["cum_hit"] + sub["cum_bb"]) / cum_pa
        sub[f"vs{hand}_hr_rate"] = sub["cum_hr"] / cum_ab
        sub[f"vs{hand}_k_rate"]  = sub["cum_kk"] / cum_pa
        sub[f"vs{hand}_pa"]      = sub["cum_ab"] + sub["cum_bb"]

        # min_pa 미달 → NaN
        mask = sub[f"vs{hand}_pa"] < min_pa
        for stat in ["avg", "obp", "hr_rate", "k_rate"]:
            sub.loc[mask, f"vs{hand}_{stat}"] = np.nan

        keep_cols = ["game_id", "batter_pcode",
                     f"vs{hand}_avg", f"vs{hand}_obp",
                     f"vs{hand}_hr_rate", f"vs{hand}_k_rate", f"vs{hand}_pa"]
        records.append(sub[keep_cols])

    left_df  = records[0].set_index(["game_id", "batter_pcode"])
    right_df = records[1].set_index(["game_id", "batter_pcode"])
    result = left_df.join(right_df, how="outer").reset_index()

    # PA 단위 중복 → 경기당 1행으로 dedup (shift(1) 기준 첫 번째 값 = 경기 시작 전 누적값)
    result = result.groupby(["game_id", "batter_pcode"], as_index=False).first()

    logger.info("타자 스플릿 생성: %d행", len(result))
    return result


# ---------------------------------------------------------------------------
# 좌우 스플릿 — 투수
# ---------------------------------------------------------------------------
def build_pitcher_split_stats(
    pa_df: pd.DataFrame,
    player_map_df: pd.DataFrame,
    min_bf: int = 20,
) -> pd.DataFrame:
    """투수의 vs_L / vs_R 스플릿 사전 기록.

    Args:
        pa_df: plate_appearances.csv.
        player_map_df: player_id_map.csv (타자 bat_hand 포함).
        min_bf: 스플릿 기준 최소 상대 타석 수.
    """
    df = pa_df.copy()
    df["pitcher_pcode"] = df.get("pitcher_pcode_norm", df.get("pitcher_pcode", "")).astype(str)
    df["batter_pcode"]  = df.get("batter_pcode_norm", df.get("batter_pcode", "")).astype(str)

    pm = player_map_df.copy()
    pm["naver_pcode"] = pm["naver_pcode"].astype(str)
    hand_map = pm.set_index("naver_pcode")["bat_hand"].to_dict()
    df["_batter_hand"] = df["batter_pcode"].map(hand_map)

    def _normalize_hand(h) -> str | None:
        if pd.isna(h):
            return None
        s = str(h).strip().upper()
        if s in {"L", "좌", "좌타"}:
            return "L"
        if s in {"R", "우", "우타"}:
            return "R"
        if s in {"S", "양", "양타", "SWITCH"}:
            return "S"
        return None

    df["_batter_hand"] = df["_batter_hand"].map(_normalize_hand)
    df = df[df["_batter_hand"].isin(["L", "R"])].copy()

    df["_is_er"]  = df["pa_result_type"].isin(["single", "double", "triple", "home_run", "walk",
                                                "intentional_walk", "hit_by_pitch"])
    df["_is_hit"] = df["pa_result_type"].isin(["single", "double", "triple", "home_run"])
    df["_is_hr"]  = df["pa_result_type"].eq("home_run")
    df["_is_bb"]  = df["pa_result_type"].isin(["walk", "intentional_walk", "hit_by_pitch"])
    df["_is_kk"]  = df["pa_result_type"].eq("strikeout")
    df["_is_bf"]  = ~df["pa_result_type"].isin(["sac_bunt"])  # BF 근사

    if "game_date" in df.columns:
        df["_date_norm"] = df["game_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    else:
        df["_date_norm"] = df["game_id"].astype(str).str[:8]
    df["season"] = df["_date_norm"].str[:4].astype(int)
    df = df.sort_values(["_date_norm", "game_id"])

    records = []
    for hand in ("L", "R"):
        sub = df[df["_batter_hand"] == hand].copy()
        grp = sub.groupby(["pitcher_pcode", "season"], group_keys=False)

        for col, raw in [("bf", "_is_bf"), ("hit", "_is_hit"),
                         ("hr", "_is_hr"), ("bb", "_is_bb"), ("kk", "_is_kk")]:
            sub[f"cum_{col}"] = grp[raw].apply(
                lambda s: s.astype(int).cumsum().shift(1)
            ).fillna(0)

        bf_safe = sub["cum_bf"].replace(0, np.nan)
        sub[f"vs{hand}_hit_rate"] = sub["cum_hit"] / bf_safe
        sub[f"vs{hand}_hr_rate"]  = sub["cum_hr"]  / bf_safe
        sub[f"vs{hand}_bb_rate"]  = sub["cum_bb"]  / bf_safe
        sub[f"vs{hand}_k_rate"]   = sub["cum_kk"]  / bf_safe
        sub[f"vs{hand}_bf"]       = sub["cum_bf"]

        mask = sub[f"vs{hand}_bf"] < min_bf
        for stat in ["hit_rate", "hr_rate", "bb_rate", "k_rate"]:
            sub.loc[mask, f"vs{hand}_{stat}"] = np.nan

        keep_cols = ["game_id", "pitcher_pcode",
                     f"vs{hand}_hit_rate", f"vs{hand}_hr_rate",
                     f"vs{hand}_bb_rate", f"vs{hand}_k_rate", f"vs{hand}_bf"]
        records.append(sub[keep_cols])

    left_df  = records[0].set_index(["game_id", "pitcher_pcode"])
    right_df = records[1].set_index(["game_id", "pitcher_pcode"])
    result = left_df.join(right_df, how="outer").reset_index()

    # PA 단위 중복 → 경기당 1행으로 dedup
    result = result.groupby(["game_id", "pitcher_pcode"], as_index=False).first()

    logger.info("투수 스플릿 생성: %d행", len(result))
    return result


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run_recency_and_split_stats(windows: list[int] | None = None) -> None:
    """Recency + 스플릿 기록을 일괄 산출하고 processed/에 저장."""
    if windows is None:
        windows = [5, 10]

    # boxscores 로드
    b_bs = pd.read_csv(config.PROCESSED_DIR / "batter_game_boxscores.csv",
                       dtype={"game_id": str, "player_code": str}, low_memory=False)
    p_bs = pd.read_csv(config.PROCESSED_DIR / "pitcher_game_boxscores.csv",
                       dtype={"game_id": str, "pcode": str}, low_memory=False)
    pa   = pd.read_csv(config.PROCESSED_DIR / "plate_appearances.csv",
                       dtype={"game_id": str, "batter_pcode": str, "pitcher_pcode": str},
                       low_memory=False)
    pm   = pd.read_csv(config.PROCESSED_DIR / "player_id_map.csv",
                       dtype={"naver_pcode": str}, low_memory=False)

    batter_rec  = build_batter_recency_stats(b_bs, windows=windows)
    pitcher_rec = build_pitcher_recency_stats(p_bs, windows=windows)
    batter_spl  = build_batter_split_stats(pa, pm)
    pitcher_spl = build_pitcher_split_stats(pa, pm)

    saves = [
        ("batter_recency_stats.csv",  batter_rec),
        ("pitcher_recency_stats.csv", pitcher_rec),
        ("batter_split_stats.csv",    batter_spl),
        ("pitcher_split_stats.csv",   pitcher_spl),
    ]
    for fname, df in saves:
        path = config.PROCESSED_DIR / fname
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("[saved] %s  rows=%d  cols=%d", path, len(df), len(df.columns))
