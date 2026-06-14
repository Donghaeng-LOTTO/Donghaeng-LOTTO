"""Win Expectancy(WE) 및 Run Expectancy(RE) 룩업 테이블 생성.

[문2016] 방식을 빈도 기반으로 구현.
  - WE(state) = P(공격팀 승리 | 이닝, 초/말, 점수차, 아웃, 주자상태)
  - RE(outs, bases) = E(이닝 잔여 득점 | 아웃카운트, 주자상태)

데이터 누수 방지:
  - train_seasons 인자로 룩업 테이블 산출 기간을 제한.
  - 예측 대상 시즌이 포함된 데이터로 WE/RE를 계산하면 안 됨.
  - run_we_re_tables()는 기본적으로 전체 데이터로 산출하되,
    모델 학습 시에는 학습 데이터 시즌만 넘겨 별도 산출 권장.

출력:
  - processed/we_table.csv  : state_key -> we, we_n
  - processed/re_table.csv  : (outs_before, base_state_before) -> re, re_n
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# 점수차 클리핑 범위. 이 범위 밖은 표본이 희소 -> 클리핑 후 통합.
SCORE_DIFF_CLIP = 6


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------
def _norm_base_state(df: pd.DataFrame) -> pd.Series:
    """base_state_before 컬럼을 정수 문자열 '000'~'111' 형태로 정규화."""
    if "base_state_before" in df.columns:
        return df["base_state_before"].astype(str).str.zfill(3)
    cols = ["base1_before", "base2_before", "base3_before"]
    b = df[cols].fillna(0).astype(int).clip(0, 1)
    return (
        b["base1_before"].astype(str)
        + b["base2_before"].astype(str)
        + b["base3_before"].astype(str)
    )


def _clip_score_diff(s: pd.Series, clip: int = SCORE_DIFF_CLIP) -> pd.Series:
    return s.clip(-clip, clip).astype(int)


def _make_we_state_key(df: pd.DataFrame) -> pd.Series:
    """WE 룩업 키: 이닝_top/bot_점수차클립_아웃_주자상태.

    side 정규화 (어떤 데이터소스든 "top"/"bot"으로 통일):
      - plate_appearances : is_top (True=top, False=bot)
      - master            : batting_team_side ("away"->top, "home"->bot)
                            또는 is_top_bool (True=top, False=bot)
    """
    inning = pd.to_numeric(df["inning"], errors="coerce").fillna(9).clip(1, 9).astype(int)

    if "batting_team_side" in df.columns:
        # master 테이블: "away"->top, "home"->bot
        side_str = df["batting_team_side"].map({"away": "top", "home": "bot"}).fillna("unk")
    elif "home_or_away" in df.columns:
        # plate_appearances: 0=원정(top), 1=홈(bot)
        # is_top 컬럼은 CSV 파싱 버그로 일부 행에 팀코드가 섞여 있어 신뢰 불가
        side_str = pd.to_numeric(df["home_or_away"], errors="coerce").fillna(0).astype(int).map({0: "top", 1: "bot"}).fillna("unk")
    elif "is_top_bool" in df.columns:
        side_str = df["is_top_bool"].map({True: "top", False: "bot"}).fillna("unk")
    elif "is_top" in df.columns:
        s = df["is_top"]
        if s.dtype == object:
            s = s.map({"True": True, "False": False, True: True, False: False})
        side_str = s.astype(bool).map({True: "top", False: "bot"}).fillna("unk")
    else:
        side_str = pd.Series("top", index=df.index)

    _diff_raw = df["batting_score_diff_before"] if "batting_score_diff_before" in df.columns else pd.Series(0, index=df.index)
    diff = _clip_score_diff(pd.to_numeric(_diff_raw, errors="coerce").fillna(0))
    outs = pd.to_numeric(df["outs_before"], errors="coerce").fillna(0).clip(0, 2).astype(int)
    bases = _norm_base_state(df)
    return (
        inning.astype(str)
        + "_" + side_str
        + "_d" + diff.astype(str)
        + "_o" + outs.astype(str)
        + "_b" + bases
    )


# ---------------------------------------------------------------------------
# WE 테이블
# ---------------------------------------------------------------------------
def build_we_table(
    pa_df: pd.DataFrame,
    min_samples: int = 30,
    score_diff_clip: int = SCORE_DIFF_CLIP,
) -> pd.DataFrame:
    """plate_appearances 데이터로 WE 룩업 테이블을 생성한다."""
    df = pa_df.copy()

    # batting_score_diff_before 파생 (PA 데이터에 없는 경우)
    if "batting_score_diff_before" not in df.columns:
        if {"away_score_before", "home_score_before", "home_or_away"}.issubset(df.columns):
            is_away = pd.to_numeric(df["home_or_away"], errors="coerce").fillna(0).astype(int) == 0
            batting = pd.to_numeric(
                df["away_score_before"].where(is_away, df["home_score_before"]), errors="coerce"
            ).fillna(0)
            fielding = pd.to_numeric(
                df["home_score_before"].where(is_away, df["away_score_before"]), errors="coerce"
            ).fillna(0)
            df["batting_score_diff_before"] = batting - fielding

    if "batting_team_win_label" not in df.columns:
        if {"winner_team_code", "batting_team_code"}.issubset(df.columns):
            df["batting_team_win_label"] = np.where(
                df["winner_team_code"].eq(df["batting_team_code"]), 1.0,
                np.where(df["winner_team_code"].eq("DRAW"), 0.5, 0.0),
            )
        else:
            raise ValueError("batting_team_win_label 또는 winner/batting_team_code 컬럼 필요")

    df = df[df["batting_team_win_label"].notna()].copy()
    if df.empty:
        logger.warning("WE 계산용 유효 행이 없습니다.")
        return pd.DataFrame(columns=["state_key", "we", "we_n"])

    df["_we_key"] = _make_we_state_key(df)

    agg = (
        df.groupby("_we_key")["batting_team_win_label"]
        .agg(we="mean", we_n="count")
        .reset_index()
        .rename(columns={"_we_key": "state_key"})
    )
    agg.loc[agg["we_n"] < min_samples, "we"] = np.nan

    logger.info(
        "WE 테이블 생성: %d 상태, 유효(n>=% d) %d개 (%.1f%%)",
        len(agg),
        min_samples,
        agg["we"].notna().sum(),
        100 * agg["we"].notna().mean(),
    )
    return agg


# ---------------------------------------------------------------------------
# RE 테이블
# ---------------------------------------------------------------------------
def build_re_table(
    pa_df: pd.DataFrame,
    min_samples: int = 10,
) -> pd.DataFrame:
    """plate_appearances 데이터로 RE 룩업 테이블을 생성한다."""
    df = pa_df.copy()

    required = {"batting_score_before", "batting_score_after"}
    has_scores = required.issubset(df.columns)
    if not has_scores:
        if {"away_score_before", "home_score_before", "is_top"}.issubset(df.columns):
            is_top = df["is_top"].astype(bool)
            df["batting_score_before"] = np.where(is_top, df["away_score_before"], df["home_score_before"])
            df["batting_score_after"] = np.where(is_top, df["away_score_after"], df["home_score_after"])
        else:
            logger.warning("RE 계산에 필요한 득점 컬럼 없음. RE 테이블 생성 건너뜀.")
            return pd.DataFrame(columns=["outs_before", "base_state_before", "re", "re_n"])

    if "half_inning_key" in df.columns:
        df["_hi_key"] = df["half_inning_key"].astype(str)
    else:
        side = df["batting_team_side"] if "batting_team_side" in df.columns else pd.Series("unk", index=df.index)
        df["_hi_key"] = (
            df["game_id"].astype(str)
            + "_" + df["inning"].astype(str)
            + "_" + side.astype(str)
        )

    df["_pa_runs"] = (
        pd.to_numeric(df["batting_score_after"], errors="coerce")
        - pd.to_numeric(df["batting_score_before"], errors="coerce")
    ).clip(lower=0).fillna(0)

    sort_col = "pa_index_in_half_inning" if "pa_index_in_half_inning" in df.columns else "relay_no"
    if sort_col not in df.columns:
        df[sort_col] = df.groupby("_hi_key").cumcount()

    df = df.sort_values(["_hi_key", sort_col])

    hi_total = df.groupby("_hi_key")["_pa_runs"].transform("sum")
    hi_cum_before = df.groupby("_hi_key")["_pa_runs"].transform(lambda s: s.cumsum().shift(1).fillna(0))
    df["_runs_rest"] = hi_total - hi_cum_before

    df["_outs"] = pd.to_numeric(df["outs_before"], errors="coerce").fillna(0).clip(0, 2).astype(int)
    df["_bases"] = _norm_base_state(df)

    agg = (
        df.groupby(["_outs", "_bases"])["_runs_rest"]
        .agg(re="mean", re_n="count")
        .reset_index()
        .rename(columns={"_outs": "outs_before", "_bases": "base_state_before"})
    )
    agg.loc[agg["re_n"] < min_samples, "re"] = np.nan

    logger.info(
        "RE 테이블 생성: %d 상태, 유효(n>=%d) %d개",
        len(agg), min_samples, agg["re"].notna().sum(),
    )
    return agg


# ---------------------------------------------------------------------------
# 마스터 조인 헬퍼
# ---------------------------------------------------------------------------
def join_we_re(master: pd.DataFrame, we_table: pd.DataFrame, re_table: pd.DataFrame) -> pd.DataFrame:
    """마스터 테이블에 WE/RE 컬럼을 조인한다."""
    out = master.copy()

    if not we_table.empty and "state_key" in we_table.columns:
        out["_we_key"] = _make_we_state_key(out)
        we_map = we_table.set_index("state_key")[["we", "we_n"]]
        out["state_we"] = out["_we_key"].map(we_map["we"])
        out["state_we_n"] = out["_we_key"].map(we_map["we_n"])
        out = out.drop(columns=["_we_key"])

    if not re_table.empty:
        outs_col = pd.to_numeric(out["outs_before"], errors="coerce").fillna(0).clip(0, 2).astype(int)
        bases_col = _norm_base_state(out)  # "000"~"111" 문자열
        # re_table CSV는 base_state_before를 정수로 저장함(앞자리 0 손실).
        # "000"->0, "010"->10, "100"->100 이므로 str.zfill(3)으로 복원
        re_norm = re_table.copy()
        re_norm["base_state_before"] = re_norm["base_state_before"].astype(str).str.zfill(3)
        re_map = re_norm.set_index(["outs_before", "base_state_before"])["re"]
        out["state_re"] = list(zip(outs_col, bases_col))
        out["state_re"] = out["state_re"].map(re_map)

    return out


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run_we_re_tables(
    train_seasons: list[int] | None = None,
    min_we_samples: int = 10,  # 30->10: 163837 상태 공간에서 30은 유효율 1%로 너무 엄격
    min_re_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """WE/RE 테이블을 산출하고 processed/에 저장한다."""
    pa_path = config.PROCESSED_DIR / "plate_appearances.csv"
    if not pa_path.exists():
        logger.error("plate_appearances.csv 없음: %s", pa_path)
        return pd.DataFrame(), pd.DataFrame()

    pa = pd.read_csv(pa_path, dtype={"game_id": str, "batter_pcode": str}, low_memory=False)

    if train_seasons is not None:
        if "season" not in pa.columns:
            pa["season"] = (
                pa["game_id"].astype(str).str[:4].pipe(pd.to_numeric, errors="coerce")
            )
        pa = pa[pa["season"].isin(train_seasons)].copy()
        logger.info("학습 시즌 필터: %s -> %d행", train_seasons, len(pa))

    we = build_we_table(pa, min_samples=min_we_samples)
    re = build_re_table(pa, min_samples=min_re_samples)

    we_path = config.PROCESSED_DIR / "we_table.csv"
    re_path = config.PROCESSED_DIR / "re_table.csv"
    we.to_csv(we_path, index=False, encoding="utf-8-sig")
    re.to_csv(re_path, index=False, encoding="utf-8-sig")
    logger.info("[saved] WE: %s (%d행)", we_path, len(we))
    logger.info("[saved] RE: %s (%d행)", re_path, len(re))

    return we, re
