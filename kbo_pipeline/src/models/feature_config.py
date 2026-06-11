"""모델 피처 정의 및 그룹 관리.

각 피처 그룹은 독립적으로 추가/제거 가능하도록 설계.
MVP 단계에서는 MVP_FEATURES만 사용하고,
고도화 단계에서 ADVANCED_FEATURES를 추가한다.

근거 논문 매핑:
  STATE_FEATURES    → [문2016] WE 상태 정의 (이닝/초말/점수차/아웃/주자)
  WE_RE_FEATURES    → [문2016] 기대승리확률/기대득점 직접 피처
  BATTER_SEASON     → [문2013] 타자 타격확률 (200타석 미만 시 통산 보정)
  PITCHER_SEASON    → [Hirotsu2005] DERA/ERA/WHIP 기반 투수 능력
  MATCHUP_FEATURES  → [MC-MLB][Matchup2025][Hirotsu2005] 좌우 매치업
  RECENCY_FEATURES  → [Matchup2025] recency가 인게임 의사결정에 중요
  SPLIT_FEATURES    → [MC-MLB] handedness adjustment, [Hirotsu2005]
  CONTEXT_FEATURES  → 투수 피로 (domain 제안, 논문 direct 근거 약함)
  PARK_FEATURES     → [MC-MLB] park factor
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. 상황 피처 — 100% 커버리지, 누수 없음
#    근거: [문2016] WE 상태 정의
# ---------------------------------------------------------------------------
STATE_FEATURES: list[str] = [
    "inning",
    "is_top_bool",              # int(0/1) 변환 필요
    "outs_before",
    "batting_score_diff_before",
    "runners_on_before",
    "base1_before",
    "base2_before",
    "base3_before",
    "scoring_position_before",  # bool → int
    "late_clutch",              # bool → int (7회↑ & 점수차≤2)
    "is_home_batting",          # bool → int
]

# ---------------------------------------------------------------------------
# 2. WE / RE 피처 — 마르코프 기반 사전 계산 테이블 조인
#    근거: [문2016]
# ---------------------------------------------------------------------------
WE_RE_FEATURES: list[str] = [
    "state_we",     # 현재 상태의 기대승리확률 (0~1)
    "state_re",     # 현재 상태의 기대득점
]

# ---------------------------------------------------------------------------
# 3. 타자 시즌 누적 기록
#    근거: [문2013] 타격확률, [김이2016] OPS
# ---------------------------------------------------------------------------
BATTER_SEASON_FEATURES: list[str] = [
    "batter_pre_games_before",
    "batter_pre_cum_ab",
    "batter_pre_avg_before",
    "batter_pre_obp_approx_before",
    "batter_pre_slg_before",
    "batter_pre_ops_before",
    "batter_pre_cum_hr",
    "batter_pre_cum_bb",
    "batter_pre_cum_kk",
]

# ---------------------------------------------------------------------------
# 4. 투수 시즌 누적 기록
#    근거: [Hirotsu2005] DERA, [MC-MLB] ERA/WHIP
# ---------------------------------------------------------------------------
PITCHER_SEASON_FEATURES: list[str] = [
    "pitcher_pre_games_before",
    "pitcher_pre_ip_before",
    "pitcher_pre_era_before",
    "pitcher_pre_whip_before",
    "pitcher_pre_k9_before",
    "pitcher_pre_bb9_before",
    "pitcher_pre_cum_hr",
]

# ---------------------------------------------------------------------------
# 5. 매치업 피처 — 좌우 정보
#    근거: [MC-MLB][Matchup2025][Hirotsu2005]
# ---------------------------------------------------------------------------
MATCHUP_FEATURES: list[str] = [
    "same_hand_matchup",        # bool → int
    "batter_platoon_advantage", # bool → int
]

# ---------------------------------------------------------------------------
# 6. Recency 이동평균 (고도화)
#    근거: [Matchup2025] recency weights
# ---------------------------------------------------------------------------
RECENCY_FEATURES: list[str] = [
    # 타자 최근 5경기
    "batter_rec_roll5_avg",
    "batter_rec_roll5_obp",
    "batter_rec_roll5_hr_rate",
    "batter_rec_roll5_k_rate",
    # 투수 최근 5경기
    "pitcher_rec_roll5_era",
    "pitcher_rec_roll5_whip",
    "pitcher_rec_roll5_k_pct",
    "pitcher_rec_roll5_bb_pct",
    # 최근 10경기
    "batter_rec_roll10_avg",
    "batter_rec_roll10_ops",
    "pitcher_rec_roll10_era",
    "pitcher_rec_roll10_whip",
]

# ---------------------------------------------------------------------------
# 7. 좌우 스플릿 (고도화)
#    근거: [MC-MLB] handedness split, [Hirotsu2005]
# ---------------------------------------------------------------------------
SPLIT_FEATURES: list[str] = [
    "batter_vsL_avg",
    "batter_vsL_obp",
    "batter_vsR_avg",
    "batter_vsR_obp",
    "pitcher_vsL_hit_rate",
    "pitcher_vsL_k_rate",
    "pitcher_vsR_hit_rate",
    "pitcher_vsR_k_rate",
]

# ---------------------------------------------------------------------------
# 8. 투수 컨텍스트 — 피로/등판 간격 (근거 약함: domain 제안)
#    [Matchup2025]가 future work로 언급
# ---------------------------------------------------------------------------
CONTEXT_FEATURES: list[str] = [
    "pitcher_rest_days",
    "pitcher_consec_days",
    "pitcher_games_last7",
]

# ---------------------------------------------------------------------------
# 9. 파크팩터 (고도화)
#    근거: [MC-MLB] park factor
# ---------------------------------------------------------------------------
PARK_FEATURES: list[str] = [
    "park_factor_runs",
]

# ---------------------------------------------------------------------------
# 통합 그룹
# ---------------------------------------------------------------------------
MVP_FEATURES: list[str] = (
    STATE_FEATURES
    + WE_RE_FEATURES
    + BATTER_SEASON_FEATURES
    + PITCHER_SEASON_FEATURES
    + MATCHUP_FEATURES
)

ADVANCED_FEATURES: list[str] = (
    MVP_FEATURES
    + RECENCY_FEATURES
    + SPLIT_FEATURES
    + CONTEXT_FEATURES
    + PARK_FEATURES
)

# 라벨 후보
LABEL_COLS: list[str] = [
    "batting_team_win_label",   # 주 타깃: 현재 공격팀 최종 승리
    "home_win_label",
    "lotte_win_label",          # 롯데 서비스 전용
]

# WE 델타 라벨 (What-if 전용)
# WE_DELTA_LABEL = "state_we_delta"  # 아직 미계산, 모델 학습 후 엔진에서 산출

# ---------------------------------------------------------------------------
# 피처 존재 여부 점검 유틸
# ---------------------------------------------------------------------------
def filter_available_features(
    df_columns: list[str],
    feature_list: list[str],
    verbose: bool = True,
) -> list[str]:
    """DataFrame 컬럼 중 실제 존재하는 피처만 반환."""
    available = [f for f in feature_list if f in df_columns]
    missing   = [f for f in feature_list if f not in df_columns]
    if verbose and missing:
        print(f"[feature_config] 없는 피처 {len(missing)}개: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return available


def get_feature_dtype_map() -> dict[str, str]:
    """LightGBM categorical 피처 힌트용 타입 맵."""
    return {
        "is_top_bool": "int8",
        "same_hand_matchup": "int8",
        "batter_platoon_advantage": "int8",
        "scoring_position_before": "int8",
        "late_clutch": "int8",
        "is_home_batting": "int8",
        "outs_before": "int8",
        "base1_before": "int8",
        "base2_before": "int8",
        "base3_before": "int8",
        "runners_on_before": "int8",
        "inning": "int8",
    }


CATEGORICAL_FEATURES: list[str] = [
    "is_top_bool",
    "same_hand_matchup",
    "batter_platoon_advantage",
    "scoring_position_before",
    "late_clutch",
    "is_home_batting",
    "outs_before",
    "base1_before",
    "base2_before",
    "base3_before",
]
