"""
05_Pregame_Predict.py — 경기 전 승리확률 예측
선발 투수 + 상대팀 선택 → 오늘 경기 예상 WP
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
PIPELINE_DIR = BASE_DIR / "kbo_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_DIR = PIPELINE_DIR / "data" / "processed"
LOTTE_RED  = "#E31937"
LOTTE_NAVY = "#001F5B"

st.set_page_config(page_title="경기 전 예측 | LOTTE GIANTS", page_icon="⚾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#001F5B 0%,#E31937 100%);border-radius:16px;padding:28px 32px;margin-bottom:24px;color:white;}
.page-header h1{font-size:2rem;font-weight:800;margin:0;}
.page-header p{font-size:1rem;opacity:.85;margin:4px 0 0;}
.big-wp{font-size:4rem;font-weight:800;line-height:1;}
.wp-card{background:white;border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.08);text-align:center;}
.stat-row{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;}
.mini-stat{background:#F9FAFB;border-radius:8px;padding:10px 16px;flex:1;min-width:80px;text-align:center;}
.mini-val{font-size:1.2rem;font-weight:700;color:#111;}
.mini-lbl{font-size:.75rem;color:#6B7280;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-header"><h1>🎯 경기 전 승리확률 예측</h1><p>선발 투수·홈/원정·상대팀을 설정하고 AI 기반 예상 WP를 확인하세요</p></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_engine():
    from src.models.whatif_engine import WhatIfEngine
    old = os.getcwd()
    os.chdir(str(PIPELINE_DIR))
    try:
        engine = WhatIfEngine.load("lgbm_model", feature_mode="mvp")
    finally:
        os.chdir(old)
    return engine

@st.cache_data(show_spinner=False)
def load_pitcher_stats():
    df = pd.read_csv(DATA_DIR / "pitcher_pre_game_stats.csv", low_memory=False)
    df["year"] = df["game_date"].astype(str).str[:4]
    return df

@st.cache_data(show_spinner=False)
def load_batter_stats():
    df = pd.read_csv(DATA_DIR / "batter_pre_game_stats.csv", low_memory=False)
    df["year"] = df["game_date"].astype(str).str[:4]
    return df

@st.cache_data(show_spinner=False)
def load_we_table():
    df = pd.read_csv(DATA_DIR / "we_table.csv")
    return df.set_index("state_key")["we"].to_dict()


TEAM_NAMES = {
    "KT":"KT 위즈","HH":"한화 이글스","HT":"KIA 타이거즈",
    "OB":"두산 베어스","LG":"LG 트윈스","WO":"키움 히어로즈",
    "NC":"NC 다이노스","SS":"삼성 라이온즈","SK":"SSG 랜더스","NC":"NC 다이노스",
}

with st.spinner("AI 모델 로드 중…"):
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}"); st.stop()

pitcher_df = load_pitcher_stats()
batter_df  = load_batter_stats()
we_table   = load_we_table()

# ── 사이드바 설정
with st.sidebar:
    st.markdown("### ⚙️ 경기 설정")
    ref_year = st.selectbox("기준 시즌 (스탯)", [str(y) for y in range(2025,2014,-1)], index=0)
    location = st.radio("롯데 홈/원정", ["홈","원정"], horizontal=True)
    opp_code = st.selectbox("상대 팀", list(TEAM_NAMES.keys()), format_func=lambda c: TEAM_NAMES[c])
    st.markdown("---")
    st.markdown("**롯데 선발 투수**")
    lt_pitchers = pitcher_df[(pitcher_df["team_code"]=="LT") & (pitcher_df["year"]==ref_year)]["name"].unique()
    sel_lt_pitcher = st.selectbox("롯데 선발", sorted(lt_pitchers)) if len(lt_pitchers) > 0 else None
    st.markdown("**상대 선발 투수**")
    opp_pitchers = pitcher_df[(pitcher_df["team_code"]==opp_code) & (pitcher_df["year"]==ref_year)]["name"].unique()
    sel_opp_pitcher = st.selectbox("상대 선발", sorted(opp_pitchers)) if len(opp_pitchers) > 0 else None

if not sel_lt_pitcher or not sel_opp_pitcher:
    st.warning("해당 시즌 투수 데이터가 없습니다."); st.stop()

def get_pitcher_stats(name, team_code, year):
    rows = pitcher_df[(pitcher_df["name"]==name) & (pitcher_df["team_code"]==team_code) & (pitcher_df["year"]==year)]
    if rows.empty:
        return {"era":4.5,"whip":1.35,"k9":7.0,"bb9":3.5,"ip":0,"hr":0,"games":0}
    last = rows.iloc[-1]
    return {
        "era":   last.get("era_before",4.5) or 4.5,
        "whip":  last.get("whip_before",1.35) or 1.35,
        "k9":    last.get("k9_before",7.0) or 7.0,
        "bb9":   last.get("bb9_before",3.5) or 3.5,
        "ip":    last.get("ip_before",0) or 0,
        "hr":    last.get("cum_hr",0) or 0,
        "games": last.get("games_before",0) or 0,
    }

def get_team_avg_batter(team_code, year):
    rows = batter_df[(batter_df["team_code"]==team_code) & (batter_df["year"]==year)]
    if rows.empty:
        return {"ops":0.730,"avg":0.270,"obp":0.340,"slg":0.390,"hr":0,"bb":0,"kk":0,"ab":0}
    last_per_player = rows.groupby("name").last()
    last_per_player = last_per_player[last_per_player["ops_before"] > 0]
    if last_per_player.empty:
        return {"ops":0.730,"avg":0.270,"obp":0.340,"slg":0.390,"hr":0,"bb":0,"kk":0,"ab":0}
    return {
        "ops": last_per_player["ops_before"].mean(),
        "avg": last_per_player["avg_before"].mean(),
        "obp": last_per_player["obp_approx_before"].mean(),
        "slg": last_per_player["slg_before"].mean(),
        "hr":  last_per_player["cum_hr"].mean(),
        "bb":  last_per_player["cum_bb"].mean(),
        "kk":  last_per_player["cum_kk"].mean(),
        "ab":  last_per_player["cum_ab"].mean(),
    }

lt_sp  = get_pitcher_stats(sel_lt_pitcher,  "LT", ref_year)
opp_sp = get_pitcher_stats(sel_opp_pitcher, opp_code, ref_year)
lt_bat  = get_team_avg_batter("LT", ref_year)
opp_bat = get_team_avg_batter(opp_code, ref_year)

is_home = (location == "홈")
state_we_init = we_table.get(f"1_{'bot' if is_home else 'top'}_d0_o0_b000", 0.5)

# 롯데 타석 WP (롯데가 타격하는 상황)
sit_lt_bat = {
    "inning":1,"is_top_bool": 0 if is_home else 1,
    "outs_before":0,"batting_score_diff_before":0,
    "runners_on_before":0,"base1_before":0,"base2_before":0,"base3_before":0,
    "scoring_position_before":0,"late_clutch":0,"is_home_batting": int(is_home),
}
player_lt_bat = {
    "pitcher_pre_era_before":   opp_sp["era"],
    "pitcher_pre_whip_before":  opp_sp["whip"],
    "pitcher_pre_k9_before":    opp_sp["k9"],
    "pitcher_pre_bb9_before":   opp_sp["bb9"],
    "pitcher_pre_games_before": opp_sp["games"],
    "pitcher_pre_ip_before":    opp_sp["ip"],
    "pitcher_pre_cum_hr":       opp_sp["hr"],
    "batter_pre_avg_before":    lt_bat["avg"],
    "batter_pre_obp_approx_before": lt_bat["obp"],
    "batter_pre_slg_before":    lt_bat["slg"],
    "batter_pre_ops_before":    lt_bat["ops"],
    "batter_pre_games_before":  0,
    "batter_pre_cum_ab":        lt_bat["ab"],
    "batter_pre_cum_hr":        lt_bat["hr"],
    "batter_pre_cum_bb":        lt_bat["bb"],
    "batter_pre_cum_kk":        lt_bat["kk"],
    "same_hand_matchup":        0,
    "batter_platoon_advantage": 0,
    "state_we": state_we_init,
    "state_re": 0.85,
}

try:
    wp_batting = engine.predict_single(sit_lt_bat, player_lt_bat)
    # 롯데 수비 시: 상대가 타격, 롯데 WP = 1 - 상대 WP
    sit_opp_bat = {**sit_lt_bat, "is_top_bool": 1 if is_home else 0, "is_home_batting": int(not is_home)}
    player_opp_bat = {**player_lt_bat,
        "pitcher_pre_era_before":   lt_sp["era"],
        "pitcher_pre_whip_before":  lt_sp["whip"],
        "pitcher_pre_k9_before":    lt_sp["k9"],
        "pitcher_pre_bb9_before":   lt_sp["bb9"],
        "pitcher_pre_games_before": lt_sp["games"],
        "pitcher_pre_ip_before":    lt_sp["ip"],
        "pitcher_pre_cum_hr":       lt_sp["hr"],
        "batter_pre_avg_before":    opp_bat["avg"],
        "batter_pre_obp_approx_before": opp_bat["obp"],
        "batter_pre_slg_before":    opp_bat["slg"],
        "batter_pre_ops_before":    opp_bat["ops"],
        "state_we": 1 - state_we_init,
    }
    wp_fielding = 1.0 - engine.predict_single(sit_opp_bat, player_opp_bat)
    lotte_wp = round((wp_batting + wp_fielding) / 2, 4)
except Exception as e:
    st.error(f"예측 오류: {e}"); st.stop()

# ── 결과 표시
col_wp, col_detail = st.columns([1, 2])

with col_wp:
    color = LOTTE_RED if lotte_wp >= 0.5 else LOTTE_NAVY
    verdict = "우위" if lotte_wp >= 0.55 else ("박빙" if lotte_wp >= 0.45 else "열세")
    st.markdown(f"""
    <div class="wp-card">
      <div style="font-size:.9rem;color:#6B7280;font-weight:600;margin-bottom:8px">롯데 자이언츠 예상 승리확률</div>
      <div class="big-wp" style="color:{color}">{lotte_wp:.1%}</div>
      <div style="font-size:1rem;font-weight:700;color:#6B7280;margin-top:8px">{verdict}</div>
      <div style="font-size:.8rem;color:#9CA3AF;margin-top:4px">vs {TEAM_NAMES[opp_code]} | {location} | {ref_year}시즌 스탯 기준</div>
    </div>""", unsafe_allow_html=True)

    # 게이지
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=lotte_wp*100,
        number={"suffix":"%","font":{"size":28}},
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":color},
               "steps":[{"range":[0,45],"color":"#EFF6FF"},
                        {"range":[45,55],"color":"#FEF9C3"},
                        {"range":[55,100],"color":"#FEF2F2"}],
               "threshold":{"line":{"color":"gray","width":2},"thickness":.75,"value":50}},
    ))
    fig_g.update_layout(paper_bgcolor="white", height=200, margin=dict(l=20,r=20,t=20,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with col_detail:
    st.markdown("#### 선발 투수 비교")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**🔴 롯데 — {sel_lt_pitcher}**")
        st.metric("ERA",  f"{lt_sp['era']:.2f}")
        st.metric("WHIP", f"{lt_sp['whip']:.2f}")
        st.metric("K/9",  f"{lt_sp['k9']:.1f}")
        st.metric("BB/9", f"{lt_sp['bb9']:.1f}")
    with c2:
        st.markdown(f"**🔵 {TEAM_NAMES[opp_code]} — {sel_opp_pitcher}**")
        delta_era  = round(lt_sp['era']  - opp_sp['era'], 2)
        delta_whip = round(lt_sp['whip'] - opp_sp['whip'], 2)
        st.metric("ERA",  f"{opp_sp['era']:.2f}",  delta=f"{-delta_era:+.2f}", delta_color="inverse")
        st.metric("WHIP", f"{opp_sp['whip']:.2f}", delta=f"{-delta_whip:+.2f}", delta_color="inverse")
        st.metric("K/9",  f"{opp_sp['k9']:.1f}")
        st.metric("BB/9", f"{opp_sp['bb9']:.1f}")

    st.markdown("#### 팀 타선 평균 OPS 비교")
    fig_bar = go.Figure(go.Bar(
        x=["롯데 자이언츠", TEAM_NAMES[opp_code]],
        y=[lt_bat["ops"], opp_bat["ops"]],
        marker_color=[LOTTE_RED, LOTTE_NAVY],
        text=[f"{lt_bat['ops']:.3f}", f"{opp_bat['ops']:.3f}"],
        textposition="outside",
    ))
    fig_bar.update_layout(paper_bgcolor="white", plot_bgcolor="white",
        height=220, margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(gridcolor="#F3F4F6", range=[0, max(lt_bat["ops"],opp_bat["ops"])*1.2]),
        showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── 시나리오 비교
st.markdown("---")
st.markdown("### 🔄 선발 투수 교체 시나리오")
st.caption("롯데 선발 투수를 다른 투수로 바꾸면 WP가 어떻게 달라지나요?")

alt_pitchers = [p for p in sorted(lt_pitchers) if p != sel_lt_pitcher][:4]
if alt_pitchers:
    scenario_cols = st.columns(len(alt_pitchers))
    for col, alt in zip(scenario_cols, alt_pitchers):
        alt_sp = get_pitcher_stats(alt, "LT", ref_year)
        player_alt = {**player_opp_bat,
            "pitcher_pre_era_before":   alt_sp["era"],
            "pitcher_pre_whip_before":  alt_sp["whip"],
            "pitcher_pre_k9_before":    alt_sp["k9"],
            "pitcher_pre_bb9_before":   alt_sp["bb9"],
            "pitcher_pre_ip_before":    alt_sp["ip"],
            "pitcher_pre_cum_hr":       alt_sp["hr"],
        }
        try:
            alt_wp_field = 1.0 - engine.predict_single(sit_opp_bat, player_alt)
            alt_lotte_wp = round((wp_batting + alt_wp_field) / 2, 4)
            delta = alt_lotte_wp - lotte_wp
            d_color = "#059669" if delta > 0.005 else ("#DC2626" if delta < -0.005 else "#6B7280")
            col.markdown(f"""
            <div style="background:white;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.06);text-align:center">
              <div style="font-weight:700;font-size:.9rem">{alt}</div>
              <div style="font-size:1.5rem;font-weight:800;color:{LOTTE_RED};margin:8px 0">{alt_lotte_wp:.1%}</div>
              <div style="font-size:.9rem;font-weight:700;color:{d_color}">{delta*100:+.1f}%p</div>
              <div style="font-size:.75rem;color:#9CA3AF">ERA {alt_sp['era']:.2f}</div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            col.error(f"{alt} 예측 실패")
