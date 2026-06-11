"""투수 컨텍스트 피처 및 파크팩터 생성.

투수 컨텍스트:
  - pitcher_rest_days     : 직전 등판 후 경과 일수 (shift(1) 적용, 누수 없음)
  - pitcher_consec_days   : 최근 연속 등판 일수
  - pitcher_games_last7   : 최근 7일 등판 경기 수 (피로 지표 근사)

파크팩터:
  - park_factor_runs      : 구장별 득점 보정 계수 (리그 평균 대비)
    = 해당 구장 경기당 평균 득점 / 전체 평균 득점
  - [MC-MLB] 홈/원정 50% 보정: 실제 구장에서 전 경기의 절반을 치르므로
    park_factor를 그대로 쓰지 않고 0.5 블렌딩 권장.

출력:
  - processed/pitcher_context.csv
  - processed/park_factors.csv
"""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 투수 컨텍스트
# ---------------------------------------------------------------------------
def build_pitcher_context(boxscores: pd.DataFrame) -> pd.DataFrame:
    """투수 등판 간격 / 연투 / 최근 등판 빈도 계산.

    Args:
        boxscores: pitcher_game_boxscores.csv
                   필수 컬럼: game_id, game_date, pcode

    Returns:
        game_id x pcode 단위 컨텍스트 피처 DataFrame.
        모든 값은 shift(1) 기반 → 당해 경기 이전 정보만 반영.
    """
    df = boxscores.copy()
    df["pcode"] = df["pcode"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["_date_norm"] = (
        df["game_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    )
    df["_date_dt"] = pd.to_datetime(df["_date_norm"], format="%Y%m%d", errors="coerce")
    df = df.sort_values(["_date_dt", "game_id"])

    out_frames = []

    for pcode, grp in df.groupby("pcode"):
        grp = grp.copy().reset_index(drop=True)

        # 직전 등판 날짜 (shift(1): 당해 경기 제외)
        grp["_prev_date"] = grp["_date_dt"].shift(1)

        # rest_days: 직전 등판 후 경과일. 첫 등판은 NaN.
        grp["pitcher_rest_days"] = (
            (grp["_date_dt"] - grp["_prev_date"]).dt.days
        )

        # 연속 등판: 어제도 등판했으면 +1 (전날 = rest_days == 1)
        grp["pitcher_consec_days"] = 0
        consec = 0
        prev_date = None
        for i, row in grp.iterrows():
            if prev_date is not None and (row["_date_dt"] - prev_date).days == 1:
                consec += 1
            else:
                consec = 0
            grp.at[i, "pitcher_consec_days"] = consec
            prev_date = row["_date_dt"]
        # shift(1): 당해 경기 전까지의 연속 등판
        grp["pitcher_consec_days"] = grp["pitcher_consec_days"].shift(1).fillna(0).astype(int)

        # 최근 7일 등판 횟수 (당해 경기 이전 기준)
        games_last7 = []
        for _, row in grp.iterrows():
            cutoff = row["_date_dt"] - timedelta(days=7)
            count = (grp["_date_dt"].shift(1) >= cutoff).sum()
            # shift(1)이 적용된 이전 경기 날짜들을 직접 세기
            past = grp[grp["_date_dt"] < row["_date_dt"]]
            cnt = ((row["_date_dt"] - past["_date_dt"]).dt.days <= 7).sum()
            games_last7.append(cnt)
        grp["pitcher_games_last7"] = games_last7

        out_frames.append(grp[["game_id", "pcode",
                                "pitcher_rest_days",
                                "pitcher_consec_days",
                                "pitcher_games_last7"]])

    result = pd.concat(out_frames, ignore_index=True)
    # rest_days = 1 → 연투. 결측(첫 등판 등)은 14로 채움 (충분히 쉰 것으로 가정)
    result["pitcher_rest_days"] = result["pitcher_rest_days"].fillna(14).clip(0, 30).astype(int)
    logger.info("투수 컨텍스트 생성: %d행", len(result))
    return result


# ---------------------------------------------------------------------------
# 파크팩터
# ---------------------------------------------------------------------------
def build_park_factors(
    games_df: pd.DataFrame,
    min_games: int = 30,
) -> pd.DataFrame:
    """구장별 파크팩터 계산.

    park_factor_runs = (구장 경기당 총득점) / (전체 평균 경기당 총득점)
    1.0 = 리그 평균. > 1.0 = 타자 유리, < 1.0 = 투수 유리.

    [MC-MLB] 주의: 선수는 홈 경기의 절반만 해당 구장에서 치르므로
    실제 모델 사용 시 park_factor를 0.5 블렌딩.

    Args:
        games_df: games.csv (stadium, away_score, home_score 포함)
        min_games: 최소 경기 수. 미달 구장은 1.0(중립) 반환.
    """
    df = games_df.copy()
    df["away_score"] = pd.to_numeric(df.get("away_score", 0), errors="coerce").fillna(0)
    df["home_score"] = pd.to_numeric(df.get("home_score", 0), errors="coerce").fillna(0)
    df["total_runs"] = df["away_score"] + df["home_score"]

    # 시즌 추출
    if "game_date" in df.columns:
        df["season"] = (
            df["game_date"].astype(str).str.replace(r"\D", "", regex=True).str[:4]
            .pipe(pd.to_numeric, errors="coerce").astype("Int64")
        )
    else:
        df["season"] = None

    league_avg = df["total_runs"].mean()
    if league_avg == 0:
        league_avg = 1.0

    agg = (
        df.groupby("stadium")["total_runs"]
        .agg(park_runs_mean="mean", park_game_count="count")
        .reset_index()
    )
    agg["park_factor_runs"] = agg["park_runs_mean"] / league_avg
    agg.loc[agg["park_game_count"] < min_games, "park_factor_runs"] = 1.0
    agg["park_factor_runs"] = agg["park_factor_runs"].round(4)

    logger.info("파크팩터 생성: %d 구장", len(agg))
    return agg[["stadium", "park_factor_runs", "park_game_count"]]


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run_pitcher_context_and_park_factors() -> None:
    p_bs   = pd.read_csv(config.PROCESSED_DIR / "pitcher_game_boxscores.csv",
                         dtype={"game_id": str, "pcode": str}, low_memory=False)
    games  = pd.read_csv(config.PROCESSED_DIR / "games.csv",
                         dtype={"game_id": str}, low_memory=False)

    ctx = build_pitcher_context(p_bs)
    pf  = build_park_factors(games)

    ctx_path = config.PROCESSED_DIR / "pitcher_context.csv"
    pf_path  = config.PROCESSED_DIR / "park_factors.csv"
    ctx.to_csv(ctx_path, index=False, encoding="utf-8-sig")
    pf.to_csv(pf_path,  index=False, encoding="utf-8-sig")
    logger.info("[saved] %s  rows=%d", ctx_path, len(ctx))
    logger.info("[saved] %s  rows=%d", pf_path,  len(pf))
