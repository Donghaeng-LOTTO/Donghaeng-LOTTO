"""
02_WhatIf_Analysis.py
"이 결정이 달랐다면?" — 경기 내 분기점별 승리확률 What-If 분석 페이지
"""
import sys
import os
from pathlib import Path

# kbo_pipeline 패키지 경로 등록
BASE_DIR = Path(__file__).parent.parent
PIPELINE_DIR = BASE_DIR / "kbo_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────
DATA_DIR   = PIPELINE_DIR / "data" / "processed"
MODELS_DIR = PIPELINE_DIR / "models"

LOTTE_CODE    = "LT"
LOTTE_RED     = "#E31937"
LOTTE_NAVY    = "#001F5B"
LOTTE_GRAY    = "#6B7280"
BG_CARD       = "#F8FAFC"

PA_COLS = [
    "game_id", "relay_no", "pa_index_in_game",
    "inning", "is_top", "is_top_bool",
    "batter_name", "pitcher_name",
    "result_text",
    "outs_before", "base1_before", "base2_before", "base3_before",
    "runners_on_before", "scoring_position_before",
    "batting_score_diff_before",
    "lotte_score_before", "opponent_score_before",
    "late_clutch", "is_home_batting", "is_lotte_batting", "is_lotte_fielding",
    "batting_team_code", "away_team_code", "home_team_code",
    "away_team_name", "home_team_name",
    "away_score", "home_score",
    "game_date",
    # 타자 스탯
    "batter_pre_avg_before", "batter_pre_obp_approx_before",
    "batter_pre_slg_before", "batter_pre_ops_before",
    "batter_pre_games_before", "batter_pre_cum_ab",
    "batter_pre_cum_hr", "batter_pre_cum_bb", "batter_pre_cum_kk",
    # 투수 스탯
    "pitcher_pre_era_before", "pitcher_pre_whip_before",
    "pitcher_pre_k9_before", "pitcher_pre_bb9_before",
    "pitcher_pre_games_before", "pitcher_pre_ip_before",
    "pitcher_pre_cum_hr",
    "has_pitcher_pre_stats",
    # 매치업
    "same_hand_matchup", "batter_platoon_advantage",
    # 라벨
    "lotte_win_label", "batting_team_win_label",
]


# ────────────────────────────────────────────────
# 데이터 캐시
# ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_lotte_games() -> pd.DataFrame:
    path = DATA_DIR / "games.csv"
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = df["game_date"].astype(str)
    lotte = df[
        (df["away_team_code"] == LOTTE_CODE) | (df["home_team_code"] == LOTTE_CODE)
    ].copy()
    # 완료 경기만 (2008: status_code=4, 2015+: status_code=RESULT)
    lotte = lotte[lotte["status_code"].astype(str).isin(["4", "RESULT"])]
    # 취소 경기 제외 (bool 또는 "N")
    cf = lotte["cancel_flag"].fillna(False)
    lotte = lotte[~cf.astype(str).isin(["True", "Y", "1"])]
    lotte["date_fmt"] = pd.to_datetime(lotte["game_date"], format="%Y%m%d", errors="coerce")
    lotte = lotte.dropna(subset=["date_fmt"])
    # PA 데이터는 2015년부터만 존재
    lotte = lotte[lotte["date_fmt"].dt.year >= 2015]
    lotte = lotte.sort_values("date_fmt", ascending=False)
    return lotte


@st.cache_data(show_spinner=False)
def load_we_table() -> dict:
    path = DATA_DIR / "we_table.csv"
    df = pd.read_csv(path)
    return df.set_index("state_key")["we"].to_dict()


@st.cache_data(show_spinner=False)
def load_re_table() -> dict:
    """RE 테이블: (outs_before, base_state_before) → re"""
    path = DATA_DIR / "re_table.csv"
    df = pd.read_csv(path)
    return {(int(r["outs_before"]), int(r["base_state_before"])): r["re"]
            for _, r in df.iterrows()}


def make_state_key(inning: int, is_top: bool, score_diff: int,
                   outs: int, b1: int, b2: int, b3: int) -> str:
    """WE 테이블 조인용 state_key 생성.
    형식: {inning}_{top/bot}_d{diff}_o{outs}_b{b1b2b3}
    점수차는 타석팀 기준 (-3~+3 클램프)
    """
    half   = "top" if is_top else "bot"
    diff   = max(-3, min(3, int(score_diff)))
    bases  = f"{int(b1)}{int(b2)}{int(b3)}"
    return f"{int(inning)}_{half}_d{diff}_o{int(outs)}_b{bases}"


@st.cache_data(show_spinner=False)
def load_game_pa(game_id: str) -> pd.DataFrame:
    path = DATA_DIR / "model_master_pa_eligible.csv"
    avail_cols = [c for c in PA_COLS if True]  # 나중에 필터
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in PA_COLS)
    return df[df["game_id"] == game_id].copy()


@st.cache_resource(show_spinner=False)
def load_engine():
    from src.models.whatif_engine import WhatIfEngine
    old_cwd = os.getcwd()
    os.chdir(str(PIPELINE_DIR))
    try:
        engine = WhatIfEngine.load("lgbm_model", feature_mode="mvp")
    finally:
        os.chdir(old_cwd)
    return engine


# ────────────────────────────────────────────────
# 승리확률 계산 (타석 단위)
# ────────────────────────────────────────────────
def compute_wp_timeline(pa_df: pd.DataFrame, engine) -> pd.DataFrame:
    """각 타석의 롯데 승리확률을 예측해 컬럼 추가."""
    we_table = load_we_table()
    re_table = load_re_table()
    rows = []
    for _, row in pa_df.iterrows():
        sit = {
            "inning":                    row.get("inning", 1),
            "is_top_bool":               int(bool(row.get("is_top_bool", row.get("is_top", 0)))),
            "outs_before":               row.get("outs_before", 0),
            "batting_score_diff_before": row.get("batting_score_diff_before", 0),
            "runners_on_before":         row.get("runners_on_before", 0),
            "base1_before":              row.get("base1_before", 0),
            "base2_before":              row.get("base2_before", 0),
            "base3_before":              row.get("base3_before", 0),
            "scoring_position_before":   int(bool(row.get("scoring_position_before", 0))),
            "late_clutch":               int(bool(row.get("late_clutch", 0))),
            "is_home_batting":           int(bool(row.get("is_home_batting", 0))),
        }
        # WE 테이블에서 state_we 조회 (타석팀 기준 점수차로 키 생성)
        batting_diff = row.get("batting_score_diff_before", 0)
        state_key = make_state_key(
            inning=sit["inning"],
            is_top=bool(row.get("is_top", True)),
            score_diff=batting_diff,
            outs=sit["outs_before"],
            b1=sit["base1_before"],
            b2=sit["base2_before"],
            b3=sit["base3_before"],
        )
        state_we_val = we_table.get(state_key, np.nan)

        player = {
            "pitcher_pre_era_before":    row.get("pitcher_pre_era_before", 4.5),
            "pitcher_pre_whip_before":   row.get("pitcher_pre_whip_before", 1.35),
            "pitcher_pre_k9_before":     row.get("pitcher_pre_k9_before", 7.0),
            "pitcher_pre_bb9_before":    row.get("pitcher_pre_bb9_before", 3.5),
            "pitcher_pre_games_before":  row.get("pitcher_pre_games_before", 0),
            "pitcher_pre_ip_before":     row.get("pitcher_pre_ip_before", 0),
            "pitcher_pre_cum_hr":        row.get("pitcher_pre_cum_hr", 0),
            "batter_pre_avg_before":     row.get("batter_pre_avg_before", 0.270),
            "batter_pre_obp_approx_before": row.get("batter_pre_obp_approx_before", 0.340),
            "batter_pre_slg_before":     row.get("batter_pre_slg_before", 0.390),
            "batter_pre_ops_before":     row.get("batter_pre_ops_before", 0.730),
            "batter_pre_games_before":   row.get("batter_pre_games_before", 0),
            "batter_pre_cum_ab":         row.get("batter_pre_cum_ab", 0),
            "batter_pre_cum_hr":         row.get("batter_pre_cum_hr", 0),
            "batter_pre_cum_bb":         row.get("batter_pre_cum_bb", 0),
            "batter_pre_cum_kk":         row.get("batter_pre_cum_kk", 0),
            "same_hand_matchup":         int(bool(row.get("same_hand_matchup", 0))),
            "batter_platoon_advantage":  int(bool(row.get("batter_platoon_advantage", 0))),
            "state_we":   state_we_val if not np.isnan(state_we_val) else 0.5,
            "state_re":   re_table.get(
                (int(sit["outs_before"]),
                 int(sit["base1_before"]) * 100 + int(sit["base2_before"]) * 10 + int(sit["base3_before"])),
                0.0
            ),
        }

        try:
            batting_wp = engine.predict_single(sit, player)
        except Exception:
            batting_wp = 0.5

        # 롯데가 수비팀이면 1 - wp
        is_lotte_batting = bool(row.get("is_lotte_batting", False))
        lotte_wp = batting_wp if is_lotte_batting else 1.0 - batting_wp

        rows.append({
            "pa_index": int(row.get("pa_index_in_game", 0)),
            "inning":   int(row.get("inning", 1)),
            "is_top":   bool(row.get("is_top", True)),
            "batter":   row.get("batter_name", ""),
            "pitcher":  row.get("pitcher_name", ""),
            "result":   row.get("result_text", ""),
            "lotte_wp": lotte_wp,
            "batting_diff": row.get("batting_score_diff_before", 0),
            "lotte_score": row.get("lotte_score_before", 0),
            "opp_score":   row.get("opponent_score_before", 0),
            "late_clutch": bool(row.get("late_clutch", False)),
            "is_lotte_batting": is_lotte_batting,
            "outs":     int(row.get("outs_before", 0)),
            "base1":    int(row.get("base1_before", 0)),
            "base2":    int(row.get("base2_before", 0)),
            "base3":    int(row.get("base3_before", 0)),
            "era":      row.get("pitcher_pre_era_before", None),
            "whip":     row.get("pitcher_pre_whip_before", None),
            "k9":       row.get("pitcher_pre_k9_before", None),
            "bb9":      row.get("pitcher_pre_bb9_before", None),
            "ops":      row.get("batter_pre_ops_before", None),
            "same_hand":     int(bool(row.get("same_hand_matchup", 0))),
            "platoon_adv":   int(bool(row.get("batter_platoon_advantage", 0))),
            "sit":      sit,
            "player":   player,
        })
    return pd.DataFrame(rows)


def build_whatif_candidates(actual_row: dict) -> list[dict]:
    """실제 상황 기반으로 가상 후보 3개 생성."""
    era  = actual_row.get("era") or 4.5
    whip = actual_row.get("whip") or 1.35
    k9   = actual_row.get("k9") or 7.0
    bb9  = actual_row.get("bb9") or 3.5
    same = actual_row.get("same_hand", 0)
    plat = actual_row.get("platoon_adv", 0)

    base_player = actual_row["player"].copy()
    # state_we는 상황 고정값이므로 모든 후보에 동일하게 유지됨

    # 후보 1: 에이스급 투수 (ERA -1.5, 삼진 +2)
    c1 = base_player.copy()
    c1.update({
        "pitcher_pre_era_before":  max(2.0, era - 1.5),
        "pitcher_pre_whip_before": max(0.9, whip - 0.25),
        "pitcher_pre_k9_before":   min(13.0, k9 + 2.0),
        "pitcher_pre_bb9_before":  max(1.5, bb9 - 0.8),
        "same_hand_matchup":       1,
        "batter_platoon_advantage": 0,
    })

    # 후보 2: 중간 불펜 투수 (ERA -0.8)
    c2 = base_player.copy()
    c2.update({
        "pitcher_pre_era_before":  max(2.5, era - 0.8),
        "pitcher_pre_whip_before": max(1.0, whip - 0.15),
        "pitcher_pre_k9_before":   min(12.0, k9 + 1.0),
        "pitcher_pre_bb9_before":  max(2.0, bb9 - 0.4),
        "same_hand_matchup":       same,
        "batter_platoon_advantage": plat,
    })

    # 후보 3: 플래툰 유리 매치업 (ERA 동일, 매치업만 변경)
    c3 = base_player.copy()
    c3.update({
        "same_hand_matchup":        0,       # 반대 손
        "batter_platoon_advantage": 1,       # 플래툰 불리
        "pitcher_pre_era_before":   era,
        "pitcher_pre_whip_before":  whip,
        "pitcher_pre_k9_before":    k9,
        "pitcher_pre_bb9_before":   bb9,
    })

    return [c1, c2, c3]


CANDIDATE_LABELS = ["에이스급 투수 투입", "중간 불펜 교체", "플래툰 매치업 변경"]


# ────────────────────────────────────────────────
# 스타일
# ────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .wi-header {
        background: linear-gradient(135deg, #001F5B 0%, #E31937 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: white;
    }
    .wi-header h1 { font-size: 2rem; font-weight: 800; margin: 0; }
    .wi-header p  { font-size: 1rem; opacity: 0.85; margin: 4px 0 0; }

    .score-card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        text-align: center;
        margin-bottom: 16px;
    }
    .score-card .team  { font-size: 0.85rem; color: #6B7280; font-weight: 600; }
    .score-card .score { font-size: 2.6rem; font-weight: 800; color: #111; line-height: 1; }
    .score-card .vs    { font-size: 1rem; color: #9CA3AF; }

    .pivot-card {
        background: white;
        border-left: 5px solid #E31937;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 14px;
        cursor: pointer;
    }
    .pivot-card:hover { box-shadow: 0 4px 16px rgba(227,25,55,0.18); }
    .pivot-title { font-size: 0.95rem; font-weight: 700; color: #111; }
    .pivot-sub   { font-size: 0.82rem; color: #6B7280; margin-top: 2px; }

    .wi-result-better  { background: #ECFDF5; border: 1.5px solid #10B981; border-radius: 10px; padding: 14px 18px; }
    .wi-result-neutral { background: #F9FAFB; border: 1.5px solid #D1D5DB; border-radius: 10px; padding: 14px 18px; }
    .wi-result-worse   { background: #FEF2F2; border: 1.5px solid #EF4444; border-radius: 10px; padding: 14px 18px; }
    .wi-delta-big   { font-size: 1.8rem; font-weight: 800; }
    .wi-delta-pos   { color: #059669; }
    .wi-delta-neg   { color: #DC2626; }
    .wi-delta-zero  { color: #6B7280; }

    .inning-label {
        font-size: 0.78rem; font-weight: 700;
        color: white; background: #001F5B;
        border-radius: 20px; padding: 2px 10px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────
# 차트
# ────────────────────────────────────────────────
def wp_timeline_chart(tl: pd.DataFrame, selected_pa: int | None = None) -> go.Figure:
    fig = go.Figure()

    # 50% 기준선
    fig.add_hline(y=0.5, line_dash="dot", line_color="#D1D5DB", line_width=1)
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(227,25,55,0.05)", line_width=0)

    # 승리확률 라인
    fig.add_trace(go.Scatter(
        x=tl["pa_index"], y=tl["lotte_wp"],
        mode="lines+markers",
        line=dict(color=LOTTE_RED, width=2.5),
        marker=dict(size=5, color=LOTTE_RED),
        name="롯데 승리확률",
        hovertemplate=(
            "<b>%{customdata[0]}회</b> %{customdata[1]}<br>"
            "%{customdata[2]} vs %{customdata[3]}<br>"
            "승리확률: <b>%{y:.1%}</b><br>"
            "%{customdata[4]}"
            "<extra></extra>"
        ),
        customdata=list(zip(
            tl["inning"],
            tl["is_top"].map({True: "초", False: "말"}),
            tl["batter"], tl["pitcher"],
            tl["result"].str[:30],
        )),
    ))

    # 선택된 분기점 강조
    if selected_pa is not None:
        sel = tl[tl["pa_index"] == selected_pa]
        if not sel.empty:
            fig.add_trace(go.Scatter(
                x=sel["pa_index"], y=sel["lotte_wp"],
                mode="markers",
                marker=dict(size=14, color=LOTTE_RED, symbol="circle", line=dict(color="white", width=2)),
                name="선택된 분기점",
                showlegend=False,
            ))

    # 이닝 구분선
    inning_changes = tl.groupby("inning")["pa_index"].min().reset_index()
    for _, r in inning_changes.iterrows():
        if r["inning"] > 1:
            fig.add_vline(x=r["pa_index"] - 0.5, line_dash="dot",
                          line_color="#E5E7EB", line_width=1)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_family="Inter",
        height=280,
        margin=dict(l=10, r=10, t=20, b=30),
        xaxis=dict(title="타석 순서", showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(
            title="롯데 승리확률",
            tickformat=".0%",
            range=[0, 1],
            gridcolor="#F3F4F6",
            tickfont=dict(size=11),
        ),
        legend=dict(font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


def delta_bar_chart(candidates_result: list[dict]) -> go.Figure:
    labels = [r["candidate_label"] for r in candidates_result]
    deltas = [r["delta_wp"] * 100 for r in candidates_result]
    colors = [LOTTE_RED if d > 0.5 else ("#9CA3AF" if abs(d) <= 0.5 else "#60A5FA") for d in deltas]

    fig = go.Figure(go.Bar(
        x=labels, y=deltas,
        marker_color=colors,
        text=[f"{d:+.1f}%" for d in deltas],
        textposition="outside",
        hovertemplate="%{x}<br>승률 변화: <b>%{y:+.1f}%p</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_family="Inter",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="승률 변화 (%p)",
            gridcolor="#F3F4F6",
            zeroline=True, zerolinecolor="#D1D5DB",
        ),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────────
# 피벗 포인트 선정
# ────────────────────────────────────────────────
def find_pivot_moments(tl: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """WP 변화량이 크고 후반이닝인 분기점을 선별."""
    df = tl.copy()
    df["wp_change"] = df["lotte_wp"].diff().abs().fillna(0)
    # 후반(7회+) 가중치
    df["score"] = df["wp_change"] * (1 + (df["inning"] >= 7).astype(int) * 0.5)
    # 클러치 가중치
    df["score"] = df["score"] * (1 + df["late_clutch"].astype(int) * 0.3)
    return df.nlargest(top_n, "score").sort_values("pa_index")


# ────────────────────────────────────────────────
# 기준 설명 텍스트
# ────────────────────────────────────────────────
BASE_RUNNERS = {(0,0,0):"없음",(1,0,0):"1루",(0,1,0):"2루",(0,0,1):"3루",
                (1,1,0):"1·2루",(1,0,1):"1·3루",(0,1,1):"2·3루",(1,1,1):"만루"}

def runners_text(row) -> str:
    key = (int(row.get("base1",0)), int(row.get("base2",0)), int(row.get("base3",0)))
    return BASE_RUNNERS.get(key, "?")


def format_situation(row: dict) -> str:
    inn   = int(row.get("inning", 1))
    top   = "초" if row.get("is_top") else "말"
    outs  = int(row.get("outs", 0))
    diff  = int(row.get("batting_diff", 0))
    diff_str = (f"+{diff}" if diff > 0 else str(diff)) if diff != 0 else "동점"
    run   = runners_text(row)
    return f"{inn}회 {top} | {outs}아웃 | 주자 {run} | 점수차 {diff_str}"


# ────────────────────────────────────────────────
# 메인 앱
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="경기 IF 분석 | LOTTE GIANTS",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── 헤더
st.markdown("""
<div class="wi-header">
  <h1>⚾ 경기 IF 분석</h1>
  <p>이 결정이 달랐다면? — 분기점별 승리확률 What-If 시뮬레이션</p>
</div>
""", unsafe_allow_html=True)

# ── 모델 로드
with st.spinner("AI 모델 초기화 중…"):
    try:
        engine = load_engine()
        model_ok = True
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        model_ok = False
        st.stop()

# ── 사이드바: 경기 선택
with st.sidebar:
    st.markdown("### 🗓 경기 선택")
    games_df = load_lotte_games()

    # 연도 선택 (최신 연도가 기본값)
    years = sorted(games_df["date_fmt"].dt.year.unique(), reverse=True)
    sel_year = st.selectbox("연도", years, index=0)

    year_games = games_df[games_df["date_fmt"].dt.year == sel_year]
    months = sorted(year_games["date_fmt"].dt.month.unique(), reverse=True)
    sel_month = st.selectbox("월", months, index=0, format_func=lambda m: f"{m}월")

    month_games = year_games[year_games["date_fmt"].dt.month == sel_month]
    # 날짜 목록
    dates = sorted(month_games["game_date"].unique(), reverse=True)
    date_labels = {d: f"{d[4:6]}/{d[6:8]}" for d in dates}
    sel_date = st.selectbox("날짜", dates, format_func=lambda d: date_labels[d])

    date_games = month_games[month_games["game_date"] == sel_date]

    def game_label(row):
        ls = int(row["home_score"]) if row["home_team_code"] == LOTTE_CODE else int(row["away_score"])
        os = int(row["away_score"]) if row["home_team_code"] == LOTTE_CODE else int(row["home_score"])
        opp = row["away_team_name"] if row["home_team_code"] == LOTTE_CODE else row["home_team_name"]
        loc = "홈" if row["home_team_code"] == LOTTE_CODE else "원정"
        return f"{opp} ({loc}) | 롯데 {ls} - {os}"

    game_options = {row["game_id"]: game_label(row) for _, row in date_games.iterrows()}
    if not game_options:
        st.warning("해당 날짜에 롯데 경기 없음")
        st.stop()

    sel_game_id = st.selectbox("경기", list(game_options.keys()),
                               format_func=lambda g: game_options[g])

    st.markdown("---")
    top_n_pivot = st.slider("분기점 표시 개수", 3, 10, 6)

# ── 데이터 로드
game_info = date_games[date_games["game_id"] == sel_game_id].iloc[0]
is_home = game_info["home_team_code"] == LOTTE_CODE
opp_name = game_info["away_team_name"] if is_home else game_info["home_team_name"]
lotte_score = int(game_info["home_score"]) if is_home else int(game_info["away_score"])
opp_score   = int(game_info["away_score"]) if is_home else int(game_info["home_score"])
win = lotte_score > opp_score

with st.spinner("경기 데이터 로딩 중…"):
    pa_df = load_game_pa(sel_game_id)

if pa_df.empty:
    st.warning("이 경기의 타석 데이터가 없습니다.")
    st.stop()

# ── 승리확률 타임라인 계산
with st.spinner("승리확률 계산 중…"):
    tl = compute_wp_timeline(pa_df, engine)

# ── 경기 요약
c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    st.markdown(f"""
    <div class="score-card">
      <div class="team">롯데 자이언츠</div>
      <div class="score" style="color:{LOTTE_RED}">{lotte_score}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    result_text = "✅ 승" if win else ("❌ 패" if lotte_score < opp_score else "🤝 무")
    st.markdown(f"""
    <div class="score-card" style="background:#F9FAFB">
      <div class="vs" style="font-size:1.5rem;font-weight:800">{result_text}</div>
      <div class="team" style="margin-top:4px">{date_labels[sel_date]} {"홈" if is_home else "원정"}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="score-card">
      <div class="team">{opp_name}</div>
      <div class="score" style="color:{LOTTE_NAVY}">{opp_score}</div>
    </div>""", unsafe_allow_html=True)

# ── 승리확률 타임라인
st.markdown("### 📈 롯데 승리확률 흐름")

pivot_df = find_pivot_moments(tl, top_n=top_n_pivot)

# 분기점 선택 (타임라인 위)
if not pivot_df.empty:
    pivot_options = {
        int(r["pa_index"]): f"{int(r['inning'])}회{'초' if r['is_top'] else '말'} | {r['batter']} vs {r['pitcher']}"
        for _, r in pivot_df.iterrows()
    }
    sel_pivot_pa = st.selectbox(
        "🔍 분기점 선택 (타임라인에 표시됩니다)",
        list(pivot_options.keys()),
        format_func=lambda p: pivot_options[p],
        key="pivot_select",
    )
else:
    sel_pivot_pa = None

st.plotly_chart(wp_timeline_chart(tl, selected_pa=sel_pivot_pa), use_container_width=True)

# ── 분기점 목록 & What-If
st.markdown("### 🔀 주요 분기점 — \"이 결정이 달랐다면?\"")

if pivot_df.empty:
    st.info("분석 가능한 분기점이 없습니다.")
else:
    for idx, (_, prow) in enumerate(pivot_df.iterrows()):
        pa_i = int(prow["pa_index"])
        is_selected = (pa_i == sel_pivot_pa)

        # 분기점 헤더
        border = f"border-left: 5px solid {LOTTE_RED};" if is_selected else "border-left: 5px solid #D1D5DB;"
        bg     = "background:#FFF5F7;" if is_selected else "background:white;"
        with st.expander(
            f"{'★ ' if is_selected else ''}  {int(prow['inning'])}회{'초' if prow['is_top'] else '말'} "
            f"| {prow['batter']} vs {prow['pitcher']}  "
            f"| 롯데 WP {prow['lotte_wp']:.1%}",
            expanded=is_selected,
        ):
            # 상황 정보
            sit_text = format_situation(prow)
            wp_now   = prow["lotte_wp"]
            prev_wp  = tl[tl["pa_index"] == pa_i - 1]["lotte_wp"].values
            wp_prev  = prev_wp[0] if len(prev_wp) > 0 else wp_now
            wp_delta = wp_now - wp_prev

            col_sit, col_wp = st.columns([3, 1])
            with col_sit:
                st.markdown(f"""
                <span class="inning-label">{int(prow['inning'])}회 {'초' if prow['is_top'] else '말'}</span>&nbsp;
                <small style="color:#6B7280">{sit_text}</small>
                """, unsafe_allow_html=True)
                st.caption(f"결과: {prow['result'][:60]}")
            with col_wp:
                arrow = "↑" if wp_delta > 0.005 else ("↓" if wp_delta < -0.005 else "→")
                color = "#059669" if wp_delta > 0.005 else ("#DC2626" if wp_delta < -0.005 else "#6B7280")
                st.markdown(f"""
                <div style="text-align:center">
                  <div style="font-size:1.5rem;font-weight:800;color:{LOTTE_RED}">{wp_now:.1%}</div>
                  <div style="font-size:0.9rem;color:{color};font-weight:700">{arrow} {abs(wp_delta):.1%}</div>
                  <div style="font-size:0.75rem;color:#9CA3AF">롯데 승리확률</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 투수 정보
            era  = prow.get("era")
            whip = prow.get("whip")
            k9   = prow.get("k9")
            bb9  = prow.get("bb9")
            ops  = prow.get("ops")

            col_p, col_b = st.columns(2)
            with col_p:
                st.markdown(f"**🧢 투수: {prow['pitcher']}**")
                if pd.notna(era):
                    st.metric("ERA", f"{era:.2f}")
                    cols3 = st.columns(3)
                    cols3[0].metric("WHIP", f"{whip:.2f}" if pd.notna(whip) else "—")
                    cols3[1].metric("K/9", f"{k9:.1f}" if pd.notna(k9) else "—")
                    cols3[2].metric("BB/9", f"{bb9:.1f}" if pd.notna(bb9) else "—")
                else:
                    st.caption("투수 기록 없음")
            with col_b:
                st.markdown(f"**🏏 타자: {prow['batter']}**")
                if pd.notna(ops):
                    st.metric("OPS", f"{ops:.3f}")
                else:
                    st.caption("타자 기록 없음")

            # ── What-If 계산
            st.markdown("#### 💡 만약 이 선택이 달랐다면…")

            candidates = build_whatif_candidates(dict(prow))
            with st.spinner("시뮬레이션 계산 중…"):
                try:
                    result = engine.predict_best_candidate(
                        situation=prow["sit"],
                        actual_features=prow["player"],
                        candidates=candidates,
                        labels=CANDIDATE_LABELS,
                    )

                    actual_wp_batting = result["actual_wp"]
                    # 롯데 WP 방향 보정
                    if not prow["is_lotte_batting"]:
                        actual_wp_lotte = 1.0 - actual_wp_batting
                        cand_results = [
                            {**r, "candidate_label": r["label"],
                             "wp_lotte": 1.0 - r["wp"],
                             "delta_lotte": -r["delta_wp"]}
                            for r in result["results"]
                        ]
                    else:
                        actual_wp_lotte = actual_wp_batting
                        cand_results = [
                            {**r, "candidate_label": r["label"],
                             "wp_lotte": r["wp"],
                             "delta_lotte": r["delta_wp"]}
                            for r in result["results"]
                        ]

                    # 결과 카드
                    for cr in cand_results:
                        delta = cr["delta_lotte"]
                        wp_l  = cr["wp_lotte"]
                        if delta > 0.005:
                            card_cls = "wi-result-better"
                            delta_cls = "wi-delta-pos"
                            icon = "✅"
                        elif delta < -0.005:
                            card_cls = "wi-result-worse"
                            delta_cls = "wi-delta-neg"
                            icon = "⚠️"
                        else:
                            card_cls = "wi-result-neutral"
                            delta_cls = "wi-delta-zero"
                            icon = "➡️"

                        st.markdown(f"""
                        <div class="{card_cls}" style="margin-bottom:10px">
                          <span style="font-weight:700;font-size:0.95rem">{icon} {cr['candidate_label']}</span>
                          <span style="float:right;font-size:0.85rem;color:#6B7280">롯데 WP: {wp_l:.1%}</span>
                          <div class="wi-delta-big {delta_cls}">{delta*100:+.1f}%p</div>
                          <div style="font-size:0.8rem;color:#6B7280">승리확률 변화 (실제 대비)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 델타 바 차트
                    bar_data = [{"candidate_label": cr["candidate_label"],
                                 "delta_wp": cr["delta_lotte"]} for cr in cand_results]
                    st.plotly_chart(delta_bar_chart(bar_data), use_container_width=True)

                    # 최선 선택 요약
                    best = max(cand_results, key=lambda x: x["delta_lotte"])
                    if best["delta_lotte"] > 0.005:
                        st.success(
                            f"**최선 선택:** {best['candidate_label']} → "
                            f"롯데 승리확률 **{best['delta_lotte']*100:+.1f}%p** 향상"
                        )
                    elif all(cr["delta_lotte"] <= 0.005 for cr in cand_results):
                        st.info("이 상황에서는 실제 선택이 최선이었거나, 시뮬레이션 후보들과 큰 차이가 없습니다.")

                except Exception as e:
                    st.error(f"시뮬레이션 오류: {e}")

# ── 하단 요약
st.markdown("---")
final_wp = tl["lotte_wp"].iloc[-1] if not tl.empty else 0.5
st.markdown(f"""
<div style="background:white;border-radius:14px;padding:20px 24px;
            box-shadow:0 2px 12px rgba(0,0,0,0.07);text-align:center;margin-top:8px">
  <div style="font-size:0.9rem;color:#6B7280;font-weight:600">경기 최종 롯데 승리확률 (AI 예측)</div>
  <div style="font-size:2.4rem;font-weight:800;color:{'#059669' if win else '#DC2626'}">{final_wp:.1%}</div>
  <div style="font-size:0.85rem;color:#9CA3AF">실제 결과: {"✅ 승" if win else "❌ 패" if lotte_score < opp_score else "🤝 무"}</div>
</div>
""", unsafe_allow_html=True)
