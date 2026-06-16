"""
03_Player_Analysis.py — 선수 개인 분석 페이지
타자/투수별 시즌 누적 스탯, 트렌드, 상대팀별 성적
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
PIPELINE_DIR = BASE_DIR / "kbo_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

DATA_DIR = PIPELINE_DIR / "data" / "processed"
LOTTE_RED  = "#E31937"
LOTTE_NAVY = "#001F5B"

st.set_page_config(page_title="선수 분석 | LOTTE GIANTS", page_icon="⚾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#001F5B 0%,#E31937 100%);border-radius:16px;padding:28px 32px;margin-bottom:24px;color:white;}
.page-header h1{font-size:2rem;font-weight:800;margin:0;}
.page-header p{font-size:1rem;opacity:.85;margin:4px 0 0;}
.stat-card{background:white;border-radius:12px;padding:18px 22px;box-shadow:0 2px 10px rgba(0,0,0,.06);text-align:center;}
.stat-val{font-size:2rem;font-weight:800;color:#E31937;}
.stat-lbl{font-size:.82rem;color:#6B7280;font-weight:600;margin-top:2px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-header"><h1>👤 선수 개인 분석</h1><p>타자·투수 시즌 누적 스탯, 트렌드, 상대팀별 성적</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_batter_stats():
    df = pd.read_csv(DATA_DIR / "batter_pre_game_stats.csv", low_memory=False)
    df["year"] = df["game_date"].astype(str).str[:4]
    return df[df["team_code"] == "LT"]

@st.cache_data(show_spinner=False)
def load_pitcher_stats():
    df = pd.read_csv(DATA_DIR / "pitcher_pre_game_stats.csv", low_memory=False)
    df["year"] = df["game_date"].astype(str).str[:4]
    return df[df["team_code"] == "LT"]

@st.cache_data(show_spinner=False)
def load_pa_for_player():
    cols = ["game_id","game_date","batter_name","pitcher_name","away_team_name","home_team_name",
            "is_lotte_batting","lotte_win_label","batter_pre_ops_before","pitcher_pre_era_before",
            "away_team_code","home_team_code","pa_result_type"]
    df = pd.read_csv(DATA_DIR / "model_master_pa_eligible.csv", low_memory=False, usecols=cols)
    df["year"] = df["game_date"].astype(str).str[:4]
    return df


# ── 사이드바
with st.sidebar:
    st.markdown("### ⚙️ 선수 선택")
    player_type = st.radio("포지션", ["타자", "투수"], horizontal=True)
    years_avail = [str(y) for y in range(2025, 2014, -1)]
    sel_year = st.selectbox("시즌", years_avail, index=0)

batter_df  = load_batter_stats()
pitcher_df = load_pitcher_stats()

if player_type == "타자":
    year_data = batter_df[batter_df["year"] == sel_year]
    player_list = sorted(year_data["name"].dropna().unique())
    if not player_list:
        st.warning("해당 시즌 데이터 없음"); st.stop()
    with st.sidebar:
        sel_player = st.selectbox("선수", player_list)

    # 선택 선수 전체 연도 데이터
    p_data = batter_df[batter_df["name"] == sel_player].copy()
    p_data = p_data.sort_values("game_date")

    # 최신 스탯
    latest = p_data[p_data["year"] == sel_year]
    if latest.empty:
        st.warning("해당 시즌 기록 없음"); st.stop()
    last = latest.iloc[-1]

    # 핵심 스탯 카드
    c1,c2,c3,c4,c5 = st.columns(5)
    stats = [
        ("타율", f"{last.get('avg_before', 0):.3f}"),
        ("OPS",  f"{last.get('ops_before', 0):.3f}"),
        ("홈런",  f"{int(last.get('cum_hr', 0))}"),
        ("타점",  f"{int(last.get('cum_rbi', 0))}"),
        ("볼넷",  f"{int(last.get('cum_bb', 0))}"),
    ]
    for col, (lbl, val) in zip([c1,c2,c3,c4,c5], stats):
        col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 시즌 트렌드", "📊 연도별 비교", "🆚 상대팀 성적"])

    with tab1:
        st.markdown(f"#### {sel_player} — {sel_year} 시즌 OPS·타율 흐름")
        season_data = p_data[p_data["year"] == sel_year].copy()
        season_data["game_no"] = range(1, len(season_data)+1)
        if len(season_data) < 2:
            st.info("데이터 부족")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=season_data["game_no"], y=season_data["ops_before"],
                name="OPS", line=dict(color=LOTTE_RED, width=2.5), mode="lines"))
            fig.add_trace(go.Scatter(x=season_data["game_no"], y=season_data["avg_before"],
                name="타율", line=dict(color=LOTTE_NAVY, width=2, dash="dot"), mode="lines"))
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
                margin=dict(l=10,r=10,t=20,b=20),
                yaxis=dict(gridcolor="#F3F4F6"),
                xaxis=dict(title="경기 순서", showgrid=False),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(f"#### {sel_player} — 연도별 주요 지표")
        yearly = p_data.groupby("year").last().reset_index()
        yearly = yearly[yearly["ops_before"].notna()]
        if yearly.empty:
            st.info("연도별 데이터 없음")
        else:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=yearly["year"], y=yearly["ops_before"],
                name="OPS", marker_color=LOTTE_RED, opacity=0.85))
            fig2.add_trace(go.Scatter(x=yearly["year"], y=yearly["avg_before"],
                name="타율", yaxis="y2", line=dict(color=LOTTE_NAVY, width=2.5), mode="lines+markers"))
            fig2.update_layout(
                paper_bgcolor="white", plot_bgcolor="white", height=300,
                margin=dict(l=10,r=10,t=20,b=20),
                yaxis=dict(title="OPS", gridcolor="#F3F4F6"),
                yaxis2=dict(title="타율", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.1),
                barmode="group")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        pa_df = load_pa_for_player()
        batter_pa = pa_df[(pa_df["batter_name"] == sel_player) & (pa_df["year"] == sel_year) & (pa_df["is_lotte_batting"] == True)]
        if batter_pa.empty:
            st.info("해당 시즌 타석 기록 없음")
        else:
            def get_opp(row):
                if row["away_team_code"] == "LT":
                    return row["home_team_name"]
                return row["away_team_name"]
            batter_pa = batter_pa.copy()
            batter_pa["상대팀"] = batter_pa.apply(get_opp, axis=1)
            grp = batter_pa.groupby("상대팀").agg(
                타석=("pa_result_type","count"),
                승리경기=("lotte_win_label", lambda x: (x==1.0).sum()),
            ).reset_index()
            grp["승률"] = (grp["승리경기"] / grp["타석"] * 100).round(1)
            fig3 = px.bar(grp.sort_values("타석", ascending=False),
                x="상대팀", y="타석", color="승률",
                color_continuous_scale=["#1E3A8A","#E31937"],
                text="타석", title=f"{sel_player} 상대팀별 출전 타석 & 팀 승률")
            fig3.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
                margin=dict(l=10,r=10,t=40,b=20), coloraxis_colorbar=dict(title="팀 승률(%)"))
            st.plotly_chart(fig3, use_container_width=True)

else:  # 투수
    year_data = pitcher_df[pitcher_df["year"] == sel_year]
    player_list = sorted(year_data["name"].dropna().unique())
    if not player_list:
        st.warning("해당 시즌 데이터 없음"); st.stop()
    with st.sidebar:
        sel_player = st.selectbox("선수", player_list)

    p_data = pitcher_df[pitcher_df["name"] == sel_player].copy()
    p_data = p_data.sort_values("game_date")
    latest = p_data[p_data["year"] == sel_year]
    if latest.empty:
        st.warning("해당 시즌 기록 없음"); st.stop()
    last = latest.iloc[-1]

    c1,c2,c3,c4,c5 = st.columns(5)
    stats = [
        ("ERA",   f"{last.get('era_before', 0):.2f}" if pd.notna(last.get('era_before')) else "—"),
        ("WHIP",  f"{last.get('whip_before', 0):.2f}" if pd.notna(last.get('whip_before')) else "—"),
        ("K/9",   f"{last.get('k9_before', 0):.1f}"  if pd.notna(last.get('k9_before'))  else "—"),
        ("BB/9",  f"{last.get('bb9_before', 0):.1f}" if pd.notna(last.get('bb9_before')) else "—"),
        ("이닝",  f"{last.get('ip_before', 0):.1f}"),
    ]
    for col, (lbl, val) in zip([c1,c2,c3,c4,c5], stats):
        col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 시즌 트렌드", "📊 연도별 비교", "🆚 상대팀 성적"])

    with tab1:
        season_data = p_data[p_data["year"] == sel_year].copy()
        season_data["game_no"] = range(1, len(season_data)+1)
        if len(season_data) < 2:
            st.info("데이터 부족")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=season_data["game_no"], y=season_data["era_before"],
                name="ERA", line=dict(color=LOTTE_RED, width=2.5), mode="lines"))
            fig.add_trace(go.Scatter(x=season_data["game_no"], y=season_data["whip_before"],
                name="WHIP", line=dict(color=LOTTE_NAVY, width=2, dash="dot"), mode="lines"))
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
                margin=dict(l=10,r=10,t=20,b=20),
                yaxis=dict(gridcolor="#F3F4F6", title="ERA / WHIP"),
                xaxis=dict(title="등판 순서", showgrid=False),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        yearly = p_data.groupby("year").last().reset_index()
        yearly = yearly[yearly["era_before"].notna()]
        if yearly.empty:
            st.info("연도별 데이터 없음")
        else:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=yearly["year"], y=yearly["era_before"],
                name="ERA", marker_color=LOTTE_RED, opacity=0.85))
            fig2.add_trace(go.Scatter(x=yearly["year"], y=yearly["k9_before"],
                name="K/9", yaxis="y2", line=dict(color=LOTTE_NAVY, width=2.5), mode="lines+markers"))
            fig2.update_layout(
                paper_bgcolor="white", plot_bgcolor="white", height=300,
                margin=dict(l=10,r=10,t=20,b=20),
                yaxis=dict(title="ERA", gridcolor="#F3F4F6"),
                yaxis2=dict(title="K/9", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        pa_df = load_pa_for_player()
        pitcher_pa = pa_df[(pa_df["pitcher_name"] == sel_player) & (pa_df["year"] == sel_year) & (pa_df["is_lotte_batting"] == False)]
        if pitcher_pa.empty:
            st.info("해당 시즌 등판 기록 없음")
        else:
            pitcher_pa = pitcher_pa.copy()
            pitcher_pa["상대팀"] = pitcher_pa.apply(
                lambda r: r["away_team_name"] if r["away_team_code"] != "LT" else r["home_team_name"], axis=1)
            grp = pitcher_pa.groupby("상대팀").agg(
                타석허용=("pa_result_type","count"),
                ERA=("pitcher_pre_era_before","last"),
            ).reset_index()
            fig3 = px.bar(grp.sort_values("타석허용", ascending=False),
                x="상대팀", y="타석허용", color="ERA",
                color_continuous_scale=["#10B981","#E31937"],
                text="타석허용", title=f"{sel_player} 상대팀별 등판 타석")
            fig3.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
                margin=dict(l=10,r=10,t=40,b=20))
            st.plotly_chart(fig3, use_container_width=True)
