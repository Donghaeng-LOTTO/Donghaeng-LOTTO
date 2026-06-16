"""
04_Season_Dashboard.py — 시즌 대시보드
롯데 시즌 승패 기록, 월별 성적, 상대팀별 승률 매트릭스, 이닝별 득실점
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "kbo_pipeline" / "data" / "processed"
LOTTE_RED  = "#E31937"
LOTTE_NAVY = "#001F5B"
LOTTE_CODE = "LT"

st.set_page_config(page_title="시즌 대시보드 | LOTTE GIANTS", page_icon="⚾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#001F5B 0%,#E31937 100%);border-radius:16px;padding:28px 32px;margin-bottom:24px;color:white;}
.page-header h1{font-size:2rem;font-weight:800;margin:0;}
.page-header p{font-size:1rem;opacity:.85;margin:4px 0 0;}
.kpi{background:white;border-radius:12px;padding:18px 16px;box-shadow:0 2px 10px rgba(0,0,0,.06);text-align:center;}
.kpi-val{font-size:2.2rem;font-weight:800;color:#E31937;line-height:1;}
.kpi-lbl{font-size:.8rem;color:#6B7280;font-weight:600;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-header"><h1>📊 시즌 대시보드</h1><p>롯데 자이언츠 시즌별 승패 흐름, 상대팀 매트릭스, 이닝별 패턴</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_games():
    df = pd.read_csv(DATA_DIR / "games.csv", low_memory=False)
    df["game_date"] = df["game_date"].astype(str)
    df["year"] = df["game_date"].str[:4]
    df["month"] = df["game_date"].str[4:6]
    df = df[df["status_code"].astype(str).isin(["4", "RESULT"])]
    df = df[~df["cancel_flag"].astype(str).isin(["True", "Y", "1"])]
    lt = df[(df["away_team_code"] == LOTTE_CODE) | (df["home_team_code"] == LOTTE_CODE)].copy()
    lt = lt[lt["year"] >= "2015"]

    def result(row):
        if row["home_team_code"] == LOTTE_CODE:
            ls, os = row["home_score"], row["away_score"]
            opp = row["away_team_name"]
            loc = "홈"
        else:
            ls, os = row["away_score"], row["home_score"]
            opp = row["home_team_name"]
            loc = "원정"
        if ls > os:   r = "승"
        elif ls < os: r = "패"
        else:          r = "무"
        return pd.Series({"lotte_score": ls, "opp_score": os, "opp_name": opp, "location": loc, "result": r})

    extra = lt.apply(result, axis=1)
    lt = pd.concat([lt.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return lt


@st.cache_data(show_spinner=False)
def load_inning_scores():
    path = DATA_DIR / "scoreboard_inning_scores.csv"
    df = pd.read_csv(path, low_memory=False)
    return df


# ── 사이드바
games = load_games()
years = sorted(games["year"].unique(), reverse=True)

with st.sidebar:
    st.markdown("### ⚙️ 시즌 선택")
    sel_year = st.selectbox("시즌", years, index=0)

gdf = games[games["year"] == sel_year].copy()

# ── KPI 카드
wins   = (gdf["result"] == "승").sum()
losses = (gdf["result"] == "패").sum()
draws  = (gdf["result"] == "무").sum()
total  = len(gdf)
decided = wins + losses
wr = wins / decided if decided > 0 else 0
rs = gdf["lotte_score"].sum()
ra = gdf["opp_score"].sum()

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    ("승률", f"{wr:.3f}"),
    ("승", str(wins)),
    ("패", str(losses)),
    ("무", str(draws)),
    ("득점합", str(int(rs))),
    ("실점합", str(int(ra))),
]
for col, (lbl, val) in zip([c1,c2,c3,c4,c5,c6], kpis):
    col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📅 월별 성적", "📈 누적 승률 추이", "🆚 상대팀 매트릭스", "🏟️ 홈/원정 비교"])

with tab1:
    st.markdown(f"#### {sel_year} 시즌 월별 승·패·무")
    monthly = gdf.groupby("month")["result"].value_counts().unstack(fill_value=0).reset_index()
    for col in ["승","패","무"]:
        if col not in monthly.columns:
            monthly[col] = 0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["승"], name="승", marker_color=LOTTE_RED))
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["패"], name="패", marker_color=LOTTE_NAVY))
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly.get("무", 0), name="무", marker_color="#9CA3AF"))
    fig.update_layout(barmode="group", paper_bgcolor="white", plot_bgcolor="white",
        height=320, margin=dict(l=10,r=10,t=20,b=20),
        xaxis=dict(title="월", showgrid=False),
        yaxis=dict(title="경기 수", gridcolor="#F3F4F6"),
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(f"#### {sel_year} 시즌 누적 승률 변화")
    gdf_sorted = gdf.sort_values("game_date").copy()
    gdf_sorted["cum_win"]  = (gdf_sorted["result"] == "승").cumsum()
    gdf_sorted["cum_loss"] = (gdf_sorted["result"] == "패").cumsum()
    gdf_sorted["cum_decided"] = gdf_sorted["cum_win"] + gdf_sorted["cum_loss"]
    gdf_sorted["cum_wr"] = gdf_sorted["cum_win"] / gdf_sorted["cum_decided"].clip(1)
    gdf_sorted["game_no"] = range(1, len(gdf_sorted)+1)

    fig2 = go.Figure()
    fig2.add_hline(y=0.5, line_dash="dot", line_color="#D1D5DB", line_width=1)
    fig2.add_trace(go.Scatter(x=gdf_sorted["game_no"], y=gdf_sorted["cum_wr"],
        mode="lines", line=dict(color=LOTTE_RED, width=2.5), name="누적 승률",
        hovertemplate="경기 %{x}<br>누적 승률: %{y:.3f}<extra></extra>"))
    fig2.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=320,
        margin=dict(l=10,r=10,t=20,b=20),
        yaxis=dict(title="누적 승률", tickformat=".0%", gridcolor="#F3F4F6", range=[0,1]),
        xaxis=dict(title="경기 순서", showgrid=False))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown(f"#### {sel_year} 상대팀별 승·패 현황")
    vs = gdf.groupby(["opp_name","result"]).size().unstack(fill_value=0).reset_index()
    for col in ["승","패","무"]:
        if col not in vs.columns: vs[col] = 0
    vs["승률"] = (vs["승"] / (vs["승"] + vs["패"]).clip(1) * 100).round(1)
    vs = vs.sort_values("승률", ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=vs["opp_name"], y=vs["승"], name="승", marker_color=LOTTE_RED))
    fig3.add_trace(go.Bar(x=vs["opp_name"], y=vs["패"], name="패", marker_color=LOTTE_NAVY))
    fig3.add_trace(go.Scatter(x=vs["opp_name"], y=vs["승률"], name="승률(%)",
        yaxis="y2", mode="lines+markers", line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=8)))
    fig3.update_layout(barmode="stack", paper_bgcolor="white", plot_bgcolor="white",
        height=340, margin=dict(l=10,r=10,t=20,b=20),
        yaxis=dict(title="경기 수", gridcolor="#F3F4F6"),
        yaxis2=dict(title="승률(%)", overlaying="y", side="right", showgrid=False, range=[0,100]),
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

    # 히트맵: 연도 × 상대팀 승률
    st.markdown("#### 연도별 상대팀 승률 히트맵 (2015~)")
    all_vs = games.groupby(["year","opp_name","result"]).size().unstack(fill_value=0).reset_index()
    for c in ["승","패"]:
        if c not in all_vs.columns: all_vs[c] = 0
    all_vs["승률"] = (all_vs["승"] / (all_vs["승"]+all_vs["패"]).clip(1)*100).round(1)
    pivot = all_vs.pivot_table(index="opp_name", columns="year", values="승률", aggfunc="mean")
    fig4 = px.imshow(pivot, color_continuous_scale=["#1E3A8A","white","#E31937"],
        zmin=0, zmax=100, aspect="auto",
        labels=dict(color="승률(%)"), title="상대팀 × 연도 승률")
    fig4.update_layout(paper_bgcolor="white", height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown(f"#### {sel_year} 홈 vs 원정 비교")
    loc_grp = gdf.groupby(["location","result"]).size().unstack(fill_value=0).reset_index()
    for c in ["승","패","무"]:
        if c not in loc_grp.columns: loc_grp[c] = 0
    loc_grp["승률"] = (loc_grp["승"] / (loc_grp["승"]+loc_grp["패"]).clip(1)*100).round(1)

    col_h, col_a = st.columns(2)
    for loc_name, col in [("홈", col_h), ("원정", col_a)]:
        row = loc_grp[loc_grp["location"] == loc_name]
        if row.empty:
            col.info(f"{loc_name} 데이터 없음")
            continue
        r = row.iloc[0]
        w, l, d = int(r.get("승",0)), int(r.get("패",0)), int(r.get("무",0))
        wr_loc = r["승률"]
        fig_pie = go.Figure(go.Pie(
            labels=["승","패","무"], values=[w,l,d],
            marker_colors=[LOTTE_RED, LOTTE_NAVY, "#9CA3AF"],
            hole=0.55, textinfo="label+value"))
        fig_pie.update_layout(
            title=dict(text=f"{loc_name} ({wr_loc:.1f}%)", font=dict(size=16, color="#111827")),
            paper_bgcolor="white", height=260, margin=dict(l=10,r=10,t=40,b=10),
            showlegend=False)
        col.plotly_chart(fig_pie, use_container_width=True)

    # 홈/원정 득실점 평균
    loc_score = gdf.groupby("location").agg(
        평균득점=("lotte_score","mean"),
        평균실점=("opp_score","mean"),
    ).reset_index()
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=loc_score["location"], y=loc_score["평균득점"], name="평균 득점", marker_color=LOTTE_RED))
    fig5.add_trace(go.Bar(x=loc_score["location"], y=loc_score["평균실점"], name="평균 실점", marker_color=LOTTE_NAVY))
    fig5.update_layout(barmode="group", paper_bgcolor="white", plot_bgcolor="white",
        height=260, margin=dict(l=10,r=10,t=20,b=20),
        yaxis=dict(title="평균 득/실점", gridcolor="#F3F4F6"),
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig5, use_container_width=True)
