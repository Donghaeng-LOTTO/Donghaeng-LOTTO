"""타석 직전 상태(점수/아웃/주자) 재구성.

전략:
1) currentGameState가 신뢰 가능하면(경기 내 nonzero 값이 존재) batter_start(8) 시점 상태 사용
2) 아니면 문자중계 텍스트 기반으로 half-inning 단위 상태 재구성 (MVP 수준)
3) 재구성 결과를 record scoreBoard 최종 점수와 비교해 검증

불확실한 이벤트는 state_parse_status / parse_warning 컬럼에 기록한다.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# currentGameState 신뢰도 판단
# ------------------------------------------------------------
def current_game_state_is_live(events_df: pd.DataFrame) -> bool:
    """과거 경기는 currentGameState가 전부 0으로 오는 경우가 있다.
    경기 전체에서 out/base/score 중 nonzero가 하나라도 있으면 live로 본다."""
    if events_df.empty:
        return False
    cols = ["out", "base1", "base2", "base3", "home_score", "away_score"]

    existing_cols = [c for c in cols if c in events_df.columns]

    if not existing_cols:
        return False

    # json에 숫자 0이 아니라 문자 "0"으로 들어오기 때문에
    sub = (
        events_df[existing_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # sub = events_df[cols].fillna(0)
    return bool((sub != 0).any().any())


# ------------------------------------------------------------
# 텍스트 기반 상태 전이
# ------------------------------------------------------------
RUNNER_BASE_RE = re.compile(r"([123])루\s*주자")
ADVANCE_TO_RE = re.compile(r"([123])루까지|([123])루\s*진루|([123])루에서\s*세이프")

# 결과 타입 -> (배터 도달 베이스, 추가 아웃 수)
RESULT_BATTER_BASE = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,        # 4 = 득점
    "walk": 1,
    "intentional_walk": 1,
    "hit_by_pitch": 1,
    "error": 1,
    "fielders_choice": 1,
}
RESULT_OUTS = {
    "strikeout": 1,
    "out": 1,
    "sac_bunt": 1,
    "sac_fly": 1,
    "double_play": 2,
    "fielders_choice": 0,  # 주자 아웃은 runner 텍스트에서 잡는다 (보수적으로 0)
}


def _count_home_in(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return text.count("홈인")


def _runner_outs_in_text(text: str) -> int:
    """runner 텍스트에서 '주자 ... 아웃' 패턴 수를 센다 (대략적)."""
    if not isinstance(text, str):
        return 0
    n = 0
    for seg in text.split("|"):
        if "주자" in seg and "아웃" in seg and "세이프" not in seg:
            n += 1
    return n


def _apply_runner_text(bases: list[bool], text: str) -> tuple[list[bool], int, int, bool]:
    """runner 텍스트를 베이스 상태에 적용.

    returns (new_bases, runs, runner_outs, uncertain)
    bases: [1루, 2루, 3루] 점유 여부
    """
    runs = 0
    runner_outs = 0
    uncertain = False
    if not isinstance(text, str) or not text.strip():
        return bases, 0, 0, False

    new_bases = bases.copy()

    for seg in text.split("|"):
        seg = seg.strip()
        if not seg:
            continue

        m = RUNNER_BASE_RE.search(seg)
        from_base = int(m.group(1)) if m else None

        if "홈인" in seg:
            runs += 1
            if from_base and new_bases[from_base - 1]:
                new_bases[from_base - 1] = False
            else:
                uncertain = True
            continue

        if "아웃" in seg and "세이프" not in seg and "주자" in seg:
            runner_outs += 1
            if from_base and new_bases[from_base - 1]:
                new_bases[from_base - 1] = False
            else:
                uncertain = True
            continue

        # 진루
        adv = ADVANCE_TO_RE.search(seg)
        if adv:
            to_base = int(next(g for g in adv.groups() if g))
            if from_base and new_bases[from_base - 1]:
                new_bases[from_base - 1] = False
                new_bases[to_base - 1] = True
            elif "도루" in seg and from_base:
                # 도루인데 출발 베이스가 비어있으면 그냥 도착 베이스만 채움
                new_bases[to_base - 1] = True
                uncertain = True
            else:
                new_bases[to_base - 1] = True
                uncertain = True

    return new_bases, runs, runner_outs, uncertain


def _apply_pa_result(bases: list[bool], pa_result_type: str, result_text: str
                     ) -> tuple[list[bool], int, int, bool]:
    """타석 결과를 베이스/아웃/득점에 적용 (배터 처리만).

    주자 이동은 runner 텍스트에서 별도 처리되므로 여기서는 배터만.
    단순화: 안타/볼넷 시 강제 진루는 runner 텍스트가 처리한다고 가정.
    """
    runs = 0
    outs = RESULT_OUTS.get(pa_result_type, 0)
    uncertain = False
    new_bases = bases.copy()

    batter_base = RESULT_BATTER_BASE.get(pa_result_type)
    if batter_base == 4:
        runs += 1  # 배터 본인 득점. 주자 득점은 runner 텍스트의 홈인으로 집계
    elif batter_base:
        if new_bases[batter_base - 1]:
            # 자리가 차 있으면 강제진루가 누락된 것 -> 불확실
            uncertain = True
        new_bases[batter_base - 1] = True

    if pa_result_type in ("unknown", "other"):
        # 아웃 여부를 텍스트로 추정
        if isinstance(result_text, str) and "아웃" in result_text:
            outs = 1
        uncertain = True

    return new_bases, runs, outs, uncertain


# ------------------------------------------------------------
# 메인: PA 단위 사전 상태 재구성
# ------------------------------------------------------------
def reconstruct_states_for_game(
    pa_df: pd.DataFrame, events_df: pd.DataFrame
) -> pd.DataFrame:
    """pa_df에 상태 컬럼을 추가해 반환.

    추가 컬럼:
    - outs_before, base1_before, base2_before, base3_before, base_state_before
    - away_score_before, home_score_before
    - state_source, state_parse_status, parse_warning
    """
    if pa_df.empty:
        return pa_df

    pa_df = pa_df.sort_values(["start_seqno"]).reset_index(drop=True).copy()
    use_cgs = current_game_state_is_live(events_df)

    if use_cgs:
        # batter_start(8) 이벤트 시점 상태를 그대로 사용
        starts = events_df[events_df["event_type"] == 8][
            ["relay_no", "seqno", "out", "base1", "base2", "base3",
             "away_score", "home_score"]
        ].sort_values("seqno").drop_duplicates(subset=["relay_no"], keep="first")

        merged = pa_df.merge(starts, on="relay_no", how="left", suffixes=("", "_cgs"))
        pa_df["outs_before"] = pd.to_numeric(merged["out"], errors="coerce")
        # base1/2/3 는 JSON에서 선수코드(non-zero=주자있음)로 오므로 clip(0,1)로 이진화
        pa_df["base1_before"] = pd.to_numeric(merged["base1"], errors="coerce").fillna(0).clip(0, 1).astype(int)
        pa_df["base2_before"] = pd.to_numeric(merged["base2"], errors="coerce").fillna(0).clip(0, 1).astype(int)
        pa_df["base3_before"] = pd.to_numeric(merged["base3"], errors="coerce").fillna(0).clip(0, 1).astype(int)
        pa_df["away_score_before"] = pd.to_numeric(merged["away_score"], errors="coerce")
        pa_df["home_score_before"] = pd.to_numeric(merged["home_score"], errors="coerce")
        # pa_df["outs_before"] = merged["out"]
        # pa_df["base1_before"] = merged["base1"].fillna(0).astype(int)
        # pa_df["base2_before"] = merged["base2"].fillna(0).astype(int)
        # pa_df["base3_before"] = merged["base3"].fillna(0).astype(int)
        # pa_df["away_score_before"] = merged["away_score"]
        # pa_df["home_score_before"] = merged["home_score"]
        # away_score_after / home_score_after: PA 마지막 이벤트 점수를 사용
        # text_reconstruction 경로와 컬럼 순서를 맞추기 위해 state_source 설정 전에 추가
        # (mode="a" CSV append는 컬럼 순서 기반으로 저장됨)
        ends = (
            events_df
            .sort_values("seqno")
            .drop_duplicates(subset=["relay_no"], keep="last")
            [["relay_no", "away_score", "home_score"]]
        )
        merged_end = pa_df[["relay_no"]].merge(ends, on="relay_no", how="left")
        pa_df["away_score_after"] = pd.to_numeric(merged_end["away_score"].values, errors="coerce")
        pa_df["home_score_after"] = pd.to_numeric(merged_end["home_score"].values, errors="coerce")
        pa_df["state_source"] = "currentGameState"
        pa_df["state_parse_status"] = "ok"
        pa_df["parse_warning"] = None
    else:
        _reconstruct_from_text(pa_df)

    pa_df["base_state_before"] = (
        pa_df["base1_before"].fillna(0).astype(int).astype(str)
        + pa_df["base2_before"].fillna(0).astype(int).astype(str)
        + pa_df["base3_before"].fillna(0).astype(int).astype(str)
    )
    return pa_df


def _reconstruct_from_text(pa_df: pd.DataFrame) -> None:
    """텍스트 기반 재구성 (in-place). half-inning 단위로 리셋."""
    outs_before = []
    b1_before, b2_before, b3_before = [], [], []
    away_before, home_before = [], []
    away_after, home_after = [], []
    statuses, warnings = [], []

    away_score = 0
    home_score = 0

    current_half = None
    bases = [False, False, False]
    outs = 0

    for _, pa in pa_df.iterrows():
        half_key = (pa["inning"], pa["home_or_away"])
        if half_key != current_half:
            current_half = half_key
            bases = [False, False, False]
            outs = 0

        # --- 사전 상태 기록 ---
        outs_before.append(outs)
        b1_before.append(int(bases[0]))
        b2_before.append(int(bases[1]))
        b3_before.append(int(bases[2]))
        away_before.append(away_score)
        home_before.append(home_score)

        # --- 상태 전이 ---
        warning_parts = []

        # 결과 전 주자 이동 (도루/폭투 등)
        bases, runs1, router1, unc1 = _apply_runner_text(
            bases, pa.get("runner_before_result_text"))
        # 타석 결과 (배터 처리)
        bases, runs2, pouts, unc2 = _apply_pa_result(
            bases, pa.get("pa_result_type"), pa.get("result_text"))
        # 결과 후 주자 이동 (홈인/진루/주자아웃)
        bases, runs3, router2, unc3 = _apply_runner_text(
            bases, pa.get("runner_after_result_text"))

        runs = runs1 + runs2 + runs3

        # 병살: 결과 자체가 2아웃을 의미하며, 결과 후 runner 텍스트의 '주자 아웃'은
        # 병살의 일부이므로 중복 합산하지 않는다.
        if pa.get("pa_result_type") == "double_play":
            pa_outs = max(2, router2)
        else:
            pa_outs = pouts + router2
        outs += router1 + pa_outs

        # 결과 텍스트 자체의 홈인도 반영 (예: '...로 출루, 3루주자 홈인'이 result에 있는 경우)
        extra_runs = _count_home_in(pa.get("result_text")) \
            - _count_home_in(pa.get("runner_after_result_text") or "") * 0
        # result_text 안 홈인은 runner 텍스트와 중복될 수 있어 보수적으로 미반영하되 기록만
        if extra_runs > 0 and runs == 0 and pa.get("pa_result_type") != "home_run":
            runs += extra_runs
            warning_parts.append("runs_from_result_text")

        # 홈런 시 베이스 클리어 + 주자 전원 득점 처리 보정
        if pa.get("pa_result_type") == "home_run":
            runners_on = sum(bases)
            # runner 텍스트에 홈인이 다 적혀 있으면 위에서 집계됨.
            # 홈인 텍스트가 부족하면 보정.
            expected = runners_on
            counted = runs3
            if counted < expected:
                runs += (expected - counted)
                warning_parts.append("hr_runner_runs_corrected")
            bases = [False, False, False]

        if unc1 or unc2 or unc3:
            warning_parts.append("base_uncertain")
        if outs > 3:
            warning_parts.append(f"outs_overflow_{outs}")
            outs = 3

        batting_side = batting_side_from_home_or_away(pa.get("home_or_away"))

        if batting_side == "away":
            away_score += runs
        elif batting_side == "home":
            home_score += runs
        else:
            warning_parts.append("unknown_batting_side")
        # if pa["home_or_away"] in ("1", 1, "home", "HOME"):
        #     home_score += runs
        # else:
        #     away_score += runs

        away_after.append(away_score)
        home_after.append(home_score)

        statuses.append("ok" if not warning_parts else "warn")
        warnings.append(",".join(warning_parts) if warning_parts else None)

    pa_df["outs_before"] = outs_before
    pa_df["base1_before"] = b1_before
    pa_df["base2_before"] = b2_before
    pa_df["base3_before"] = b3_before
    pa_df["away_score_before"] = away_before
    pa_df["home_score_before"] = home_before
    pa_df["away_score_after"] = away_after
    pa_df["home_score_after"] = home_after
    pa_df["state_source"] = "text_reconstruction"
    pa_df["state_parse_status"] = statuses
    pa_df["parse_warning"] = warnings


# ------------------------------------------------------------
# 정규화 함수 추가
# ------------------------------------------------------------

def normalize_half(value) -> str:
    """
    Naver relay 기준:
    0 = 초 공격 = away batting
    1 = 말 공격 = home batting
    """
    if value is None:
        return "unknown"

    s = str(value).strip()

    # CSV에서 0.0으로 들어오는 경우까지 방어
    if s in {"0", "0.0"}:
        return "top"

    if s in {"1", "1.0"}:
        return "bottom"

    return "unknown"


def batting_side_from_home_or_away(value) -> str:
    half = normalize_half(value)

    if half == "top":
        return "away"

    if half == "bottom":
        return "home"

    return "unknown"


def safe_int(value, default=None):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    s = str(value).strip()

    if s == "" or s.lower() in {"nan", "none", "null"}:
        return default

    try:
        return int(float(s))
    except ValueError:
        return default


def compare_score(reconstructed_away, reconstructed_home, record_away, record_home) -> bool | None:
    ra = safe_int(reconstructed_away)
    rh = safe_int(reconstructed_home)
    aa = safe_int(record_away)
    ah = safe_int(record_home)

    if None in [ra, rh, aa, ah]:
        return None

    return (ra == aa) and (rh == ah)


# ------------------------------------------------------------
# 검증: 재구성 최종 점수 vs record scoreBoard
# ------------------------------------------------------------

def validate_final_score(
    pa_df: pd.DataFrame, away_score_record, home_score_record
) -> dict:
    """경기 단위 검증 결과 dict 반환 (quality report용)."""
    if pa_df.empty:
        return {
            "reconstructed_away": None,
            "reconstructed_home": None,
            "record_away": safe_int(away_score_record),
            "record_home": safe_int(home_score_record),
            "away_score_gap": None,
            "home_score_gap": None,
            "score_match": None,
        }

    pa_df = pa_df.sort_values("start_seqno").copy()

    # 가장 정확한 건 재구성 후 점수
    if {"away_score_after", "home_score_after"}.issubset(pa_df.columns):
        rec_away = pa_df["away_score_after"].dropna().iloc[-1]
        rec_home = pa_df["home_score_after"].dropna().iloc[-1]
    else:
        # fallback: 기존 방식
        rec_away = pa_df["away_score_before"].max()
        rec_home = pa_df["home_score_before"].max()

    rec_away = safe_int(rec_away)
    rec_home = safe_int(rec_home)
    record_away = safe_int(away_score_record)
    record_home = safe_int(home_score_record)

    score_match = compare_score(
        rec_away,
        rec_home,
        record_away,
        record_home,
    )

    away_score_gap = (
        None if rec_away is None or record_away is None
        else rec_away - record_away
    )

    home_score_gap = (
        None if rec_home is None or record_home is None
        else rec_home - record_home
    )

    return {
        "reconstructed_away": rec_away,
        "reconstructed_home": rec_home,
        "record_away": record_away,
        "record_home": record_home,
        "away_score_gap": away_score_gap,
        "home_score_gap": home_score_gap,
        "score_match": score_match,
    }



# def validate_final_score(
#     pa_df: pd.DataFrame, away_score_record, home_score_record
# ) -> dict:
#     """경기 단위 검증 결과 dict 반환 (quality report용)."""
#     if pa_df.empty:
#         return {
#             "reconstructed_away": None,
#             "reconstructed_home": None,
#             "record_away": away_score_record,
#             "record_home": home_score_record,
#             "away_score_gap": None,
#             "home_score_gap": None,
#             "score_match": None,
#         }

    # last = pa_df.iloc[-1]
    # 마지막 PA 사전 점수 + 마지막 PA 득점은 별도 계산이 필요하므로
    # 간단히: scoring PA들의 합으로 최종 점수 추정이 어려워
    # state_source가 currentGameState면 마지막 이벤트 점수를 신뢰,
    # text 재구성이면 (사전점수 + 마지막 PA 득점)을 추정해야 한다.
    # MVP: pa_df에 남은 *_score_before 최댓값 기반 근사 + 허용오차 보고.
    # rec_away = pa_df["away_score_before"].max()
    # rec_home = pa_df["home_score_before"].max()

    # record_away = safe_int(away_score_record)
    # record_home = safe_int(home_score_record)

    # score_match = compare_score(
    #     rec_away,
    #     rec_home,
    #     record_away,
    #     record_home,
    # )

    # away_score_gap = (
    #     None if rec_away is None or record_away is None
    #     else rec_away - record_away
    # )

    # home_score_gap = (
    #     None if rec_home is None or record_home is None
    #     else rec_home - record_home
    # )

    # return {
    #     "reconstructed_away": rec_away,
    #     "reconstructed_home": rec_home,
    #     "record_away": record_away,
    #     "record_home": record_home,
    #     "away_score_gap": away_score_gap,
    #     "home_score_gap": home_score_gap,
    #     "score_match": score_match,
    # }

    # def _match(a, b):
    #     if a is None or b is None or pd.isna(a) or pd.isna(b):
    #         return None
    #     return bool(int(a) <= int(b) and int(b) - int(a) <= 4)
    #     # 마지막 이닝 득점이 *_before에 반영되지 않으므로 약간의 차이는 허용

    # return {
    #     "reconstructed_away": rec_away,
    #     "reconstructed_home": rec_home,
    #     "record_away": away_score_record,
    #     "record_home": home_score_record,
    #     "score_match": (
    #         _match(rec_away, away_score_record) and _match(rec_home, home_score_record)
    #         if _match(rec_away, away_score_record) is not None
    #         and _match(rec_home, home_score_record) is not None
    #         else None
    #     ),
    # }
