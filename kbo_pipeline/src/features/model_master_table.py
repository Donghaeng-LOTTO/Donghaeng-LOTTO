"""모델 학습용 PA 단위 마스터 테이블 생성.

입력(processed):
- plate_appearances.csv             : 타석 단위 상태/결과 중심 테이블
- games_detail.csv                  : 경기 메타/최종 승패
- player_id_map.csv                 : 선수 손잡이/포지션/외부 ID 매핑
- batter_pre_game_stats.csv         : 타자 경기 전 누적 기록
- pitcher_pre_game_stats.csv        : 투수 경기 전 누적 기록
- score_validation.csv              : 재구성 점수 검증 결과
- dataset_quality_report.csv        : 경기 단위 품질 지표

출력(processed):
- model_master_pa.csv               : 전체 PA 마스터 테이블
- model_master_pa_eligible.csv      : 1차 모델 학습 권장 행만 필터링한 테이블

핵심 원칙:
- 한 행 = 한 타석(plate appearance)
- 경기 종료 후 기록(boxscore)은 직접 피처로 넣지 않고, shift(1) 된 pre-game 누적 기록만 조인한다.
- 승패 타깃은 home/away뿐 아니라 현재 공격팀 관점(batting_team_win_label)을 함께 만든다.
- state_parse_status, score_match 등 품질 정보를 남겨 모델 학습 시 필터링할 수 있게 한다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

ID_DTYPES = {
    "game_id": str,
    "batter_pcode": str,
    "batter_pcode_norm": str,
    "pitcher_pcode": str,
    "pitcher_pcode_norm": str,
    "player_code": str,
    "pcode": str,
    "naver_pcode": str,
    "away_starting_pitcher_pcode": str,
    "home_starting_pitcher_pcode": str,
}


# ---------------------------------------------------------------------------
# 기본 유틸
# ---------------------------------------------------------------------------
def _read_processed_csv(filename: str, required: bool = True) -> pd.DataFrame:
    path = config.PROCESSED_DIR / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"필수 파일이 없습니다: {path}")
        logger.warning("optional processed file not found: %s", path)
        return pd.DataFrame()

    # 존재하지 않는 dtype key는 pandas가 무시하지 않으므로 실제 컬럼을 먼저 읽지 않고도
    # 문제가 없도록 공통 id 컬럼만 dtype 지정한다.
    return pd.read_csv(path, dtype=ID_DTYPES, low_memory=False)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_bool_series(s: pd.Series) -> pd.Series:
    true_values = {"true", "1", "1.0", "t", "y", "yes"}
    false_values = {"false", "0", "0.0", "f", "n", "no"}

    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return pd.NA

    return s.map(convert).astype("boolean")


def _score_match_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.Series(dtype="boolean")
    return _to_bool_series(s)


def _base_state_from_cols(df: pd.DataFrame) -> pd.Series:
    base_cols = ["base1_before", "base2_before", "base3_before"]
    if not set(base_cols).issubset(df.columns):
        return pd.Series(pd.NA, index=df.index, dtype="string")

    b = df[base_cols].fillna(0).astype(int).clip(0, 1)
    return (
        b["base1_before"].astype(str)
        + b["base2_before"].astype(str)
        + b["base3_before"].astype(str)
    ).astype("string")


def _rename_non_keys(df: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    rename_map = {col: f"{prefix}_{col}" for col in df.columns if col not in keys}
    return df.rename(columns=rename_map)


def _side_from_top(is_top: pd.Series) -> pd.Series:
    is_top_np = is_top.astype(object).eq(True)
    return pd.Series(
        np.select(
            [is_top_np, is_top.astype(object).eq(False)],
            ["away", "home"],
            default=pd.NA,
        ),
        index=is_top.index,
        dtype="string",
    )


def _opposite_side(side: pd.Series) -> pd.Series:
    side_obj = side.astype(object)
    return pd.Series(
        np.select(
            [side_obj.eq("away"), side_obj.eq("home")],
            ["home", "away"],
            default=pd.NA,
        ),
        index=side.index,
        dtype="string",
    )


# ---------------------------------------------------------------------------
# 조인 로직
# ---------------------------------------------------------------------------
def _merge_games_detail(pa: pd.DataFrame, games_detail: pd.DataFrame) -> pd.DataFrame:
    gd_cols = [
        "game_id", "game_date", "game_time", "stadium",
        "away_team_name", "away_team_full_name",
        "home_team_name", "home_team_full_name",
        "away_starting_pitcher_pcode", "home_starting_pitcher_pcode",
        "status_code", "cancel_flag",
    ]
    keep = [c for c in gd_cols if c in games_detail.columns and (c == "game_id" or c not in pa.columns)]
    gd = games_detail[keep].drop_duplicates("game_id").copy()
    return pa.merge(gd, on="game_id", how="left")


def _merge_score_validation(pa: pd.DataFrame, score_validation: pd.DataFrame) -> pd.DataFrame:
    if score_validation.empty:
        pa["score_match"] = pd.NA
        return pa

    sv = score_validation.copy()
    keep = [
        c for c in [
            "game_id", "score_match", "record_away", "record_home",
            "reconstructed_away", "reconstructed_home",
            "away_score_gap", "home_score_gap",
        ]
        if c in sv.columns
    ]
    sv = sv[keep].drop_duplicates("game_id")
    sv = _rename_non_keys(sv, keys=["game_id"], prefix="score_validation")
    out = pa.merge(sv, on="game_id", how="left")
    if "score_validation_score_match" in out.columns:
        out["score_match"] = _score_match_series(out["score_validation_score_match"])
    else:
        out["score_match"] = pd.NA
    return out


def _merge_quality_report(pa: pd.DataFrame, quality_report: pd.DataFrame) -> pd.DataFrame:
    if quality_report.empty:
        return pa

    q = quality_report.copy()
    keep = [
        c for c in [
            "game_id", "has_record_raw", "has_games_detail", "has_relay_raw",
            "relay_inning_files", "n_relay_events", "n_plate_appearances",
            "pa_count_plausible", "state_warn_ratio",
        ]
        if c in q.columns
    ]
    q = q[keep].drop_duplicates("game_id")
    q = _rename_non_keys(q, keys=["game_id"], prefix="quality")
    return pa.merge(q, on="game_id", how="left")


def _merge_player_map(pa: pd.DataFrame, player_map: pd.DataFrame) -> pd.DataFrame:
    if player_map.empty:
        return pa

    pm_cols = [
        "naver_pcode", "throw_hand", "bat_hand", "hit_type_raw", "pos_name",
        "source_role", "kaggle_id", "kaggle_handedness_raw", "kaggle_source_role",
        "match_status", "match_method", "match_confidence", "name_candidate_count",
    ]
    pm_cols = [c for c in pm_cols if c in player_map.columns]
    pm = player_map[pm_cols].drop_duplicates("naver_pcode").copy()

    batter_pm = _rename_non_keys(pm, keys=["naver_pcode"], prefix="batter")
    batter_pm = batter_pm.rename(columns={"naver_pcode": "batter_pcode_norm"})
    out = pa.merge(batter_pm, on="batter_pcode_norm", how="left")

    pitcher_pm = _rename_non_keys(pm, keys=["naver_pcode"], prefix="pitcher")
    pitcher_pm = pitcher_pm.rename(columns={"naver_pcode": "pitcher_pcode_norm"})
    out = out.merge(pitcher_pm, on="pitcher_pcode_norm", how="left")
    return out


def _merge_batter_pre_stats(pa: pd.DataFrame, batter_pre: pd.DataFrame) -> pd.DataFrame:
    if batter_pre.empty:
        pa["has_batter_pre_stats"] = False
        return pa

    bp = batter_pre.copy()
    bp["batter_pcode_norm"] = bp["player_code"].astype(str)
    bp["batting_team_code"] = bp["team_code"].astype(str)
    bp = bp.drop(columns=["player_code", "team_code"], errors="ignore")
    keys = ["game_id", "batter_pcode_norm", "batting_team_code"]
    bp = _rename_non_keys(bp, keys=keys, prefix="batter_pre")

    out = pa.merge(bp, on=keys, how="left")
    out["has_batter_pre_stats"] = out.get("batter_pre_games_before", pd.Series(np.nan, index=out.index)).notna()
    out["batter_has_history"] = pd.to_numeric(
        out.get("batter_pre_games_before", 0), errors="coerce"
    ).fillna(0).gt(0)
    return out


def _merge_pitcher_pre_stats(pa: pd.DataFrame, pitcher_pre: pd.DataFrame) -> pd.DataFrame:
    if pitcher_pre.empty:
        pa["has_pitcher_pre_stats"] = False
        return pa

    pp = pitcher_pre.copy()
    pp["pitcher_pcode_norm"] = pp["pcode"].astype(str)
    pp["fielding_team_code"] = pp["team_code"].astype(str)
    pp = pp.drop(columns=["pcode", "team_code"], errors="ignore")
    keys = ["game_id", "pitcher_pcode_norm", "fielding_team_code"]
    pp = _rename_non_keys(pp, keys=keys, prefix="pitcher_pre")

    out = pa.merge(pp, on=keys, how="left")
    out["has_pitcher_pre_stats"] = out.get("pitcher_pre_games_before", pd.Series(np.nan, index=out.index)).notna()
    out["pitcher_has_history"] = pd.to_numeric(
        out.get("pitcher_pre_games_before", 0), errors="coerce"
    ).fillna(0).gt(0)
    return out


# ---------------------------------------------------------------------------
# 파생 피처/라벨
# ---------------------------------------------------------------------------
def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "relay_no", "inning", "start_seqno", "result_seqno", "end_seqno",
        "batter_order", "pitch_count", "outs_before",
        "base1_before", "base2_before", "base3_before",
        "away_score_before", "home_score_before",
        "away_score_after", "home_score_after",
        "away_score", "home_score",
    ]
    out = _to_numeric(out, numeric_cols)

    # is_top이 CSV를 거치며 문자열이 될 수 있어 다시 boolean으로 정규화한다.
    if "is_top" in out.columns:
        out["is_top_bool"] = _to_bool_series(out["is_top"])
    else:
        out["is_top_bool"] = out["home_or_away"].astype(str).isin(["0", "0.0", "away", "top"])
        out["is_top_bool"] = out["is_top_bool"].astype("boolean")

    out["batting_team_side"] = _side_from_top(out["is_top_bool"])
    out["fielding_team_side"] = _opposite_side(out["batting_team_side"])
    out["is_home_batting"] = out["batting_team_side"].eq("home")
    out["is_away_batting"] = out["batting_team_side"].eq("away")

    _is_top = out["is_top_bool"].astype(object).eq(True)
    out["batting_score_before"] = np.where(
        _is_top, out["away_score_before"], out["home_score_before"]
    )
    out["fielding_score_before"] = np.where(
        _is_top, out["home_score_before"], out["away_score_before"]
    )
    out["batting_score_after"] = np.where(
        _is_top, out["away_score_after"], out["home_score_after"]
    )
    out["fielding_score_after"] = np.where(
        _is_top, out["home_score_after"], out["away_score_after"]
    )

    out["batting_score_diff_before"] = out["batting_score_before"] - out["fielding_score_before"]
    out["batting_score_diff_after"] = out["batting_score_after"] - out["fielding_score_after"]
    out["home_score_diff_before"] = out["home_score_before"] - out["away_score_before"]
    out["score_abs_diff_before"] = out["home_score_diff_before"].abs()

    out["base_state_before"] = _base_state_from_cols(out)
    out["runners_on_before"] = (
        out[["base1_before", "base2_before", "base3_before"]]
        .fillna(0)
        .astype(int)
        .sum(axis=1)
    )
    out["is_bases_empty_before"] = out["runners_on_before"].eq(0)
    out["is_bases_loaded_before"] = out["runners_on_before"].eq(3)

    out["half_inning_key"] = (
        out["game_id"].astype(str)
        + "_" + out["inning"].astype("Int64").astype(str)
        + "_" + out["batting_team_side"].astype(str)
    )

    sort_cols = [c for c in ["game_id", "relay_no", "start_seqno", "result_seqno"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out["pa_index_in_game"] = out.groupby("game_id").cumcount() + 1
    out["pa_index_in_half_inning"] = out.groupby("half_inning_key").cumcount() + 1

    out["state_id_before"] = (
        out["inning"].astype("Int64").astype(str)
        + "_" + out["batting_team_side"].astype(str)
        + "_outs" + out["outs_before"].astype("Int64").astype(str)
        + "_bases" + out["base_state_before"].astype(str)
        + "_diff" + out["batting_score_diff_before"].astype("Int64").astype(str)
    )

    if "game_date" in out.columns:
        date_norm = out["game_date"].astype(str).str.replace(r"\D", "", regex=True)
        out["season"] = pd.to_numeric(date_norm.str[:4], errors="coerce").astype("Int64")

    return out


def _add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    winner = out.get("winner_team_code", pd.Series(pd.NA, index=out.index)).astype("string").astype(object)
    home = out.get("home_team_code", pd.Series(pd.NA, index=out.index)).astype("string").astype(object)
    away = out.get("away_team_code", pd.Series(pd.NA, index=out.index)).astype("string").astype(object)
    batting = out.get("batting_team_code", pd.Series(pd.NA, index=out.index)).astype("string").astype(object)
    fielding = out.get("fielding_team_code", pd.Series(pd.NA, index=out.index)).astype("string").astype(object)

    out["is_draw_game"] = winner.eq("DRAW")

    out["home_win_label"] = np.select(
        [winner.eq(home), winner.eq("DRAW"), winner.isna()],
        [1.0, 0.5, np.nan],
        default=0.0,
    )
    out["away_win_label"] = np.select(
        [winner.eq(away), winner.eq("DRAW"), winner.isna()],
        [1.0, 0.5, np.nan],
        default=0.0,
    )
    out["batting_team_win_label"] = np.select(
        [winner.eq(batting), winner.eq("DRAW"), winner.isna()],
        [1.0, 0.5, np.nan],
        default=0.0,
    )
    out["fielding_team_win_label"] = np.select(
        [winner.eq(fielding), winner.eq("DRAW"), winner.isna()],
        [1.0, 0.5, np.nan],
        default=0.0,
    )

    lotte_code = config.LOTTE_TEAM_CODE
    is_lotte_game = home.eq(lotte_code) | away.eq(lotte_code)
    out["lotte_win_label"] = np.where(
        ~is_lotte_game,
        np.nan,
        np.where(winner.eq(lotte_code), 1.0, np.where(winner.eq("DRAW"), 0.5, 0.0)),
    )
    return out


def _add_result_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    result = out.get("pa_result_type", pd.Series("", index=out.index)).fillna("").astype(str)
    result_text = out.get("result_text", pd.Series("", index=out.index)).fillna("").astype(str)

    out["is_single"] = result.eq("single")
    out["is_double"] = result.eq("double")
    out["is_triple"] = result.eq("triple")
    out["is_home_run"] = result.eq("home_run")
    out["is_hit"] = result.isin(["single", "double", "triple", "home_run"])
    out["is_extra_base_hit"] = result.isin(["double", "triple", "home_run"])
    out["is_walk"] = result.isin(["walk", "intentional_walk"])
    out["is_intentional_walk"] = result.eq("intentional_walk")
    out["is_hit_by_pitch"] = result.eq("hit_by_pitch")
    out["is_strikeout"] = result.eq("strikeout")
    out["is_double_play"] = result.eq("double_play")
    out["is_error_result"] = result.eq("error")
    out["is_fielders_choice"] = result.eq("fielders_choice")
    out["is_sacrifice"] = result.isin(["sac_bunt", "sac_fly"])
    out["is_sac_bunt"] = result.eq("sac_bunt")
    out["is_sac_fly"] = result.eq("sac_fly")

    # 작전성 이벤트 후보. 결과 문자열 기반이라 1차 휴리스틱으로 남긴다.
    out["strategy_bunt_candidate"] = result.eq("sac_bunt") | result_text.str.contains("번트", na=False)
    out["strategy_steal_candidate"] = result_text.str.contains("도루", na=False)
    out["strategy_squeeze_candidate"] = result_text.str.contains("스퀴즈", na=False)

    return out


def _normalize_side(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"L", "좌", "좌투", "좌타", "LEFT"}:
        return "L"
    if text in {"R", "우", "우투", "우타", "RIGHT"}:
        return "R"
    if text in {"S", "양", "양타", "SWITCH"}:
        return "S"
    return None


def _add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    batter_side = out.get("batter_bat_hand", pd.Series(pd.NA, index=out.index)).map(_normalize_side)
    pitcher_side = out.get("pitcher_throw_hand", pd.Series(pd.NA, index=out.index)).map(_normalize_side)

    out["batter_bat_side_norm"] = batter_side
    out["pitcher_throw_side_norm"] = pitcher_side
    out["same_hand_matchup"] = (
        batter_side.notna() & pitcher_side.notna() & batter_side.eq(pitcher_side)
    )

    # 매우 단순한 플래툰 휴리스틱:
    # - 우타자는 좌투 상대, 좌타자는 우투 상대일 때 advantage로 본다.
    # - 스위치히터는 별도 모델링 전까지 advantage=True 후보로 둔다.
    advantage = (
        (batter_side.eq("R") & pitcher_side.eq("L"))
        | (batter_side.eq("L") & pitcher_side.eq("R"))
        | batter_side.eq("S")
    )
    known_matchup = batter_side.notna() & pitcher_side.notna()
    out["batter_platoon_advantage"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[known_matchup, "batter_platoon_advantage"] = advantage.loc[known_matchup].astype(bool)

    return out


def _add_eligibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    core_required = [
        "game_id", "inning", "batting_team_code", "fielding_team_code",
        "outs_before", "base_state_before", "batting_score_diff_before",
        "batter_pcode_norm", "pitcher_pcode_norm", "batting_team_win_label",
    ]

    out["state_is_ok"] = out.get("state_parse_status", pd.Series("", index=out.index)).eq("ok")
    score_match = out.get("score_match", pd.Series(True, index=out.index))
    if not pd.api.types.is_bool_dtype(score_match):
        score_match = _to_bool_series(score_match)
    out["score_is_valid"] = score_match.fillna(False).astype(bool)

    out["core_fields_notna"] = out[core_required].notna().all(axis=1)
    out["outs_before_valid"] = pd.to_numeric(out["outs_before"], errors="coerce").between(0, 2)
    out["inning_valid"] = pd.to_numeric(out["inning"], errors="coerce").ge(1)

    out["model_eligible_base"] = (
        out["state_is_ok"]
        & out["score_is_valid"]
        & out["core_fields_notna"]
        & out["outs_before_valid"]
        & out["inning_valid"]
    )

    out["model_eligible_with_player_stats"] = (
        out["model_eligible_base"]
        & out.get("has_batter_pre_stats", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        & out.get("has_pitcher_pre_stats", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )

    issues = pd.Series("", index=out.index, dtype="string")

    def add_issue(mask: pd.Series, label: str) -> None:
        nonlocal issues
        mask = mask.fillna(False)
        issues = issues.mask(mask, issues + ";" + label)

    add_issue(~out["state_is_ok"], "state_not_ok")
    add_issue(~out["score_is_valid"], "score_not_valid")
    add_issue(~out["core_fields_notna"], "missing_core_field")
    add_issue(~out["outs_before_valid"], "invalid_outs_before")
    add_issue(~out["inning_valid"], "invalid_inning")
    add_issue(~out.get("has_batter_pre_stats", pd.Series(True, index=out.index)), "missing_batter_pre_stats")
    add_issue(~out.get("has_pitcher_pre_stats", pd.Series(True, index=out.index)), "missing_pitcher_pre_stats")

    out["model_exclusion_reasons"] = issues.str.lstrip(";")
    out.loc[out["model_eligible_with_player_stats"], "model_exclusion_reasons"] = ""
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def build_model_master_table() -> pd.DataFrame:
    """processed CSV들을 합쳐 PA 단위 모델 마스터 테이블을 반환한다."""
    pa = _read_processed_csv("plate_appearances.csv")
    games_detail = _read_processed_csv("games_detail.csv")
    player_map = _read_processed_csv("player_id_map.csv", required=False)
    batter_pre = _read_processed_csv("batter_pre_game_stats.csv", required=False)
    pitcher_pre = _read_processed_csv("pitcher_pre_game_stats.csv", required=False)
    score_validation = _read_processed_csv("score_validation.csv", required=False)
    quality_report = _read_processed_csv("dataset_quality_report.csv", required=False)

    pa = pa.copy()
    pa["game_id"] = pa["game_id"].astype(str)

    # norm 컬럼이 없는 과거 산출물도 처리 가능하게 보정한다.
    if "batter_pcode_norm" not in pa.columns:
        pa["batter_pcode_norm"] = pa.get("batter_pcode")
    if "pitcher_pcode_norm" not in pa.columns:
        pa["pitcher_pcode_norm"] = pa.get("pitcher_pcode")

    pa["batter_pcode_norm"] = pa["batter_pcode_norm"].astype(str)
    pa["pitcher_pcode_norm"] = pa["pitcher_pcode_norm"].astype(str)

    master = _merge_games_detail(pa, games_detail)
    master = _add_context_features(master)
    master = _add_target_labels(master)
    master = _add_result_flags(master)
    master = _merge_score_validation(master, score_validation)
    master = _merge_quality_report(master, quality_report)
    master = _merge_player_map(master, player_map)
    master = _merge_batter_pre_stats(master, batter_pre)
    master = _merge_pitcher_pre_stats(master, pitcher_pre)
    master = _add_matchup_features(master)
    master = _add_eligibility_flags(master)

    return master.reset_index(drop=True)


def run_model_master_table(
    output_path: str | Path | None = None,
    eligible_output_path: str | Path | None = None,
    write_eligible: bool = True,
) -> pd.DataFrame:
    """마스터 테이블을 생성하고 CSV로 저장한다."""
    output = Path(output_path) if output_path else config.PROCESSED_DIR / "model_master_pa.csv"
    eligible_output = (
        Path(eligible_output_path)
        if eligible_output_path
        else config.PROCESSED_DIR / "model_master_pa_eligible.csv"
    )

    master = build_model_master_table()
    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s rows=%d cols=%d", output, len(master), len(master.columns))

    if write_eligible:
        eligible_col = "model_eligible_with_player_stats"
        eligible = master[master[eligible_col]].copy()
        eligible.to_csv(eligible_output, index=False, encoding="utf-8-sig")
        logger.info(
            "[saved] %s rows=%d cols=%d eligibility=%.1f%% criteria=%s",
            eligible_output,
            len(eligible),
            len(eligible.columns),
            100 * len(eligible) / len(master) if len(master) else 0,
            eligible_col,
        )

    if not master.empty:
        logger.info("model_eligible_base:\n%s", master["model_eligible_base"].value_counts(dropna=False))
        logger.info(
            "model_eligible_with_player_stats:\n%s",
            master["model_eligible_with_player_stats"].value_counts(dropna=False),
        )
        if "model_exclusion_reasons" in master.columns:
            logger.info(
                "top exclusion reasons:\n%s",
                master.loc[~master["model_eligible_with_player_stats"], "model_exclusion_reasons"].value_counts().head(10),
            )

    return master
