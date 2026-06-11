"""model_master_table.py에 신규 피처를 추가 조인하는 확장 모듈.

기존 build_model_master_table()이 생성한 master DataFrame 위에
아래 피처들을 추가로 조인한다:
  1. WE / RE (we_table.csv, re_table.csv)
  2. Recency 이동평균 (batter_recency_stats.csv, pitcher_recency_stats.csv)
  3. 좌우 스플릿 (batter_split_stats.csv, pitcher_split_stats.csv)
  4. 투수 컨텍스트 (pitcher_context.csv)
  5. 파크팩터 (park_factors.csv)

사용법:
    from src.features.model_master_table import build_model_master_table
    from src.features.model_master_table_ext import build_extended_master_table

    master = build_model_master_table()
    extended = build_extended_master_table(master)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config
from src.features.we_re_table import join_we_re

logger = logging.getLogger(__name__)

ID_DTYPES = {
    "game_id": str,
    "batter_pcode": str,
    "batter_pcode_norm": str,
    "pitcher_pcode": str,
    "pitcher_pcode_norm": str,
    "player_code": str,
    "pcode": str,
}


def _safe_read(filename: str, required: bool = False) -> pd.DataFrame:
    path = config.PROCESSED_DIR / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"필수 파일 없음: {path}")
        logger.warning("파일 없음(건너뜀): %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=ID_DTYPES, low_memory=False)


# ---------------------------------------------------------------------------
# WE / RE 조인
# ---------------------------------------------------------------------------
def _join_we_re(master: pd.DataFrame) -> pd.DataFrame:
    we_table = _safe_read("we_table.csv")
    re_table = _safe_read("re_table.csv")
    if we_table.empty and re_table.empty:
        logger.warning("WE/RE 테이블 없음 — run_we_re_tables() 먼저 실행 필요")
        return master
    return join_we_re(master, we_table, re_table)


# ---------------------------------------------------------------------------
# Recency 조인
# ---------------------------------------------------------------------------
def _join_recency(master: pd.DataFrame) -> pd.DataFrame:
    batter_rec  = _safe_read("batter_recency_stats.csv")
    pitcher_rec = _safe_read("pitcher_recency_stats.csv")

    out = master.copy()

    if not batter_rec.empty:
        batter_rec["batter_pcode_norm"] = batter_rec["player_code"].astype(str)
        batter_rec["batting_team_code"] = batter_rec["team_code"].astype(str)
        # 기존 피처 컬럼과 충돌 방지: 모든 통계 컬럼에 접두사 추가
        stat_cols = [c for c in batter_rec.columns
                     if c not in ("game_id", "player_code", "team_code",
                                  "batter_pcode_norm", "batting_team_code",
                                  "game_date", "season")]
        rename = {c: f"batter_rec_{c}" for c in stat_cols}
        batter_rec = batter_rec.rename(columns=rename)
        keys = ["game_id", "batter_pcode_norm", "batting_team_code"]
        batter_rec = batter_rec.drop(columns=["player_code", "team_code"], errors="ignore")
        out = out.merge(batter_rec, on=keys, how="left")
        logger.info("타자 recency 조인: %d컬럼 추가", len(rename))

    if not pitcher_rec.empty:
        pitcher_rec["pitcher_pcode_norm"] = pitcher_rec["pcode"].astype(str)
        pitcher_rec["fielding_team_code"] = pitcher_rec["team_code"].astype(str)
        stat_cols = [c for c in pitcher_rec.columns
                     if c not in ("game_id", "pcode", "team_code",
                                  "pitcher_pcode_norm", "fielding_team_code",
                                  "game_date", "season")]
        rename = {c: f"pitcher_rec_{c}" for c in stat_cols}
        pitcher_rec = pitcher_rec.rename(columns=rename)
        pitcher_rec = pitcher_rec.drop(columns=["pcode", "team_code"], errors="ignore")
        keys = ["game_id", "pitcher_pcode_norm", "fielding_team_code"]
        out = out.merge(pitcher_rec, on=keys, how="left")
        logger.info("투수 recency 조인: %d컬럼 추가", len(rename))

    return out


# ---------------------------------------------------------------------------
# 스플릿 조인
# ---------------------------------------------------------------------------
def _join_splits(master: pd.DataFrame) -> pd.DataFrame:
    batter_spl  = _safe_read("batter_split_stats.csv")
    pitcher_spl = _safe_read("pitcher_split_stats.csv")

    out = master.copy()

    if not batter_spl.empty:
        batter_spl["batter_pcode_norm"] = batter_spl["batter_pcode"].astype(str)
        spl_cols = [c for c in batter_spl.columns if c not in ("game_id", "batter_pcode", "batter_pcode_norm")]
        rename = {c: f"batter_{c}" for c in spl_cols}
        batter_spl = batter_spl.rename(columns=rename)
        out = out.merge(
            batter_spl.drop(columns=["batter_pcode"], errors="ignore"),
            on=["game_id", "batter_pcode_norm"],
            how="left",
        )
        logger.info("타자 스플릿 조인: %d컬럼 추가", len(rename))

    if not pitcher_spl.empty:
        pitcher_spl["pitcher_pcode_norm"] = pitcher_spl["pitcher_pcode"].astype(str)
        spl_cols = [c for c in pitcher_spl.columns if c not in ("game_id", "pitcher_pcode", "pitcher_pcode_norm")]
        rename = {c: f"pitcher_{c}" for c in spl_cols}
        pitcher_spl = pitcher_spl.rename(columns=rename)
        out = out.merge(
            pitcher_spl.drop(columns=["pitcher_pcode"], errors="ignore"),
            on=["game_id", "pitcher_pcode_norm"],
            how="left",
        )
        logger.info("투수 스플릿 조인: %d컬럼 추가", len(rename))

    return out


# ---------------------------------------------------------------------------
# 투수 컨텍스트 조인
# ---------------------------------------------------------------------------
def _join_pitcher_context(master: pd.DataFrame) -> pd.DataFrame:
    ctx = _safe_read("pitcher_context.csv")
    if ctx.empty:
        return master
    ctx["pitcher_pcode_norm"] = ctx["pcode"].astype(str)
    ctx = ctx.drop(columns=["pcode"], errors="ignore")
    out = master.merge(ctx, on=["game_id", "pitcher_pcode_norm"], how="left")
    logger.info("투수 컨텍스트 조인 완료")
    return out


# ---------------------------------------------------------------------------
# 파크팩터 조인
# ---------------------------------------------------------------------------
def _join_park_factors(master: pd.DataFrame) -> pd.DataFrame:
    pf = _safe_read("park_factors.csv")
    if pf.empty:
        return master
    if "stadium" not in master.columns:
        logger.warning("stadium 컬럼 없음 — 파크팩터 건너뜀")
        return master
    out = master.merge(pf[["stadium", "park_factor_runs"]], on="stadium", how="left")
    out["park_factor_runs"] = out["park_factor_runs"].fillna(1.0)
    logger.info("파크팩터 조인 완료")
    return out


# ---------------------------------------------------------------------------
# 홈어드밴티지 피처
# ---------------------------------------------------------------------------
def _add_home_advantage_feature(master: pd.DataFrame) -> pd.DataFrame:
    """batting_team이 홈 팀인지 여부 → is_home_batting (이미 존재할 수 있음)."""
    out = master.copy()
    if "is_home_batting" not in out.columns:
        if "batting_team_side" in out.columns:
            out["is_home_batting"] = out["batting_team_side"].eq("home")
        elif {"batting_team_code", "home_team_code"}.issubset(out.columns):
            out["is_home_batting"] = out["batting_team_code"].eq(out["home_team_code"])
        else:
            out["is_home_batting"] = False
    out["is_home_batting"] = out["is_home_batting"].astype(int)
    return out


# ---------------------------------------------------------------------------
# 메인 확장 빌더
# ---------------------------------------------------------------------------
def build_extended_master_table(master: pd.DataFrame) -> pd.DataFrame:
    """기존 마스터 테이블에 신규 피처를 순차적으로 조인한다.

    Args:
        master: build_model_master_table()의 출력 DataFrame.

    Returns:
        신규 피처가 추가된 확장 마스터 DataFrame.
    """
    logger.info("확장 피처 조인 시작 (입력: %d행 x %d열)", len(master), len(master.columns))

    out = _join_we_re(master)
    out = _join_recency(out)
    out = _join_splits(out)
    out = _join_pitcher_context(out)
    out = _join_park_factors(out)
    out = _add_home_advantage_feature(out)

    logger.info("확장 피처 조인 완료 → %d열 (%+d열 추가)",
                len(out.columns), len(out.columns) - len(master.columns))
    return out.reset_index(drop=True)


def run_extended_master_table(
    output_path: str | None = None,
    eligible_output_path: str | None = None,
) -> pd.DataFrame:
    """확장 마스터 테이블 생성 후 CSV 저장."""
    from src.features.model_master_table import build_model_master_table

    out_path = Path(output_path) if output_path else config.PROCESSED_DIR / "model_master_pa_extended.csv"
    elig_path = Path(eligible_output_path) if eligible_output_path else config.PROCESSED_DIR / "model_master_pa_extended_eligible.csv"

    master   = build_model_master_table()
    extended = build_extended_master_table(master)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s  rows=%d  cols=%d", out_path, len(extended), len(extended.columns))

    eligible_col = "model_eligible_with_player_stats"
    if eligible_col in extended.columns:
        elig = extended[extended[eligible_col]].copy()
        elig.to_csv(elig_path, index=False, encoding="utf-8-sig")
        logger.info("[saved] %s  rows=%d  eligibility=%.1f%%",
                    elig_path, len(elig), 100 * len(elig) / len(extended) if len(extended) else 0)

    return extended
