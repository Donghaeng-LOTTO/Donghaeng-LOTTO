"""
06_Pitcher_Timing.py — 투수 교체 타이밍 분석
실제 교체 시점 분포, WP 변화, 최적 교체 타이밍 인사이트
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

st.set_page_config(page_title="투수 교체 타이밍 | LOTTE GIANTS", page_icon="⚾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.page-header{background:linear-gradient(135deg,#001F5B 0%,#E31937 100%);border-radius:16px;padding:28px 32px;margin-bottom:24px;color:white;}
.page-header h1{font-size:2rem;font-weight:800;margin:0;}
.page-header p{font-size:1rem;opacity:.85;margin:4px 0 0;}
.insight-card{background:white;border-left:4px solid #E31937;border-radius:10px;padding:14px 18px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:12px;}
.insight-title{font-weight:700;font-size:.95rem;color:#111;}
.insight-body{font-size:.85rem;color:#6B7280;margin-top:4px;}
.kpi{background:white;border-radius:12px;padding:18px 16px;box-shadow:0 2px 10px rgba(0,0,0,.06);text-align:center;}
.kpi-val{font-size:2.2rem;font-weight:800;color:#E31937;line-height:1;}
.kpi-lbl{font-size:.8rem;color:#6B7280;font-weight:600;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-header"><h1>🔀 투수 교체 타이밍 분석</html><p>실제 교체 시점의 이닝·점수·승리확률 분포와 최적 타이밍 인사이트</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_pa_changes():
    cols = [
        "game_id","game_date","inning","is_lotte_batting","has_player_change","change_text",
        "batting_score_diff_before","outs_before","base1_before","base2_before","base3_before",
        "lotte_win_label","lotte_wp_before","lotte_wp_after",
        "pitcher_pre_era_before","pitcher_pre_whip_before",
        "away_team_name","home_team_name","away_team_code","home_team_code",
    ]
    available_cols = cols
    df = pd.read_csv(DATA_DIR / "model_master_pa_eligible.csv", low_memory=False, usecols=lambda c: c in cols)
    df["year"] = df["game_date"].astype(str).str[:4]
    df["month"] = df["game_date"].astype(str).str[4:6]
    # 투수 교체 행만
    mask_change = df["has_player_change"].astype(str).isin(["True","1","true"])
    if "change_text" in df.columns:
        mask_pitcher = df["change_text"].astype(str).str.contains("투수|P:", na=False)
        df = df[mask_change & mask_pitcher].copy()
    else:
        df = df[mask_change].copy()
    return df


@st.cache_data(show_spinner=False)
def load_pa_full_wp():
    """WP 컬럼이 있으면 로드, 없으면 None"""
    cols = ["game_id","game_date","inning","is_lotte_batting",
            "lotte_wp_before","lotte_wp_after","has_player_change","change_text"]
    try:
        df = pd.read_csv(DATA_DIR / "model_master_pa_eligible.csv", low_memory=False,
                         usecols=lambda c: c in cols)
        if "lotte_wp_before" not in df.columns:
            return None
        df["year"] = df["game_date"].astype(str).str[:4]
        return df
    except Exception:
        return None


# ── 데이터 로드
with st.spinner("데이터 로딩 중…"):
    changes = load_pa_changes()
    has_wp = "lotte_wp_before" in changes.columns and changes["lotte_wp_before"].notna().sum() > 100

years = sorted(changes["year"].unique(), reverse=True)

with st.sidebar:
    st.markdown("### ⚙️ 필터")
    sel_year = st.selectbox("시즌", ["전체"] + list(years), index=0)
    batting_side = st.radio("롯데 입장", ["수비 중 (상대 타격)", "공격 중 (롯데 타격)", "전체"], index=0)

df = changes.copy()
if sel_year != "전체":
    df = df[df["year"] == sel_year]

if batting_side == "수비 중 (상대 타격)":
    df = df[df["is_lotte_batting"] == False]
elif batting_side == "공격 중 (롯데 타격)":
    df = df[df["is_lotte_batting"] == True]

if df.empty:
    st.warning("해당 조건의 교체 기록이 없습니다."); st.stop()

# ── KPI
c1,c2,c3,c4 = st.columns(4)
total_changes = len(df)
avg_inning = df["inning"].mean()
games_with_change = df["game_id"].nunique()

if has_wp and "lotte_win_label" in df.columns:
    win_rate_after_change = (df["lotte_win_label"] == 1).mean()
    c4.markdown(f'<div class="kpi"><div class="kpi-val">{win_rate_after_change:.1%}</div><div class="kpi-lbl">교체 경기 승률</div></div>', unsafe_allow_html=True)
else:
    c4.markdown(f'<div class="kpi"><div class="kpi-val">—</div><div class="kpi-lbl">교체 경기 승률</div></div>', unsafe_allow_html=True)

c1.markdown(f'<div class="kpi"><div class="kpi-val">{total_changes:,}</div><div class="kpi-lbl">총 교체 횟수</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><div class="kpi-val">{games_with_change:,}</div><div class="kpi-lbl">교체 발생 경기</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><div class="kpi-val">{avg_inning:.1f}회</div><div class="kpi-lbl">평균 교체 이닝</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📍 이닝 분포", "📊 점수 상황", "📈 WP 변화", "💡 최적 타이밍"])

with tab1:
    st.markdown("#### 이닝별 투수 교체 횟수")
    inning_cnt = df.groupby("inning").size().reset_index(name="count")
    inning_cnt = inning_cnt[inning_cnt["inning"] <= 15]

    fig = go.Figure(go.Bar(
        x=inning_cnt["inning"], y=inning_cnt["count"],
        marker_color=[LOTTE_RED if i >= 7 else LOTTE_NAVY for i in inning_cnt["inning"]],
        text=inning_cnt["count"], textposition="outside",
    ))
    fig.add_vline(x=6.5, line_dash="dash", line_color="#9CA3AF", line_width=1,
                  annotation_text="7회 이후 (불펜)", annotation_position="top right")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=320,
        margin=dict(l=10,r=10,t=20,b=20),
        xaxis=dict(title="이닝", dtick=1, showgrid=False),
        yaxis=dict(title="교체 횟수", gridcolor="#F3F4F6"),
        showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 후반 vs 전반
    early = (df["inning"] <= 6).sum()
    late  = (df["inning"] >= 7).sum()
    col_a, col_b = st.columns(2)
    col_a.metric("1~6회 교체", f"{early}회", f"{early/total_changes:.0%}")
    col_b.metric("7회~ 교체",  f"{late}회",  f"{late/total_changes:.0%}")

with tab2:
    st.markdown("#### 교체 시점 점수 상황 분포")
    if "batting_score_diff_before" in df.columns:
        df2 = df.copy()
        df2["점수차"] = df2["batting_score_diff_before"].fillna(0).astype(int).clip(-5, 5)
        score_cnt = df2.groupby("점수차").size().reset_index(name="count")

        colors = [LOTTE_RED if s > 0 else ("#9CA3AF" if s == 0 else LOTTE_NAVY) for s in score_cnt["점수차"]]
        fig2 = go.Figure(go.Bar(x=score_cnt["점수차"], y=score_cnt["count"],
            marker_color=colors, text=score_cnt["count"], textposition="outside"))
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300,
            margin=dict(l=10,r=10,t=20,b=20),
            xaxis=dict(title="타격팀 기준 점수차 (양수=리드)", showgrid=False, dtick=1),
            yaxis=dict(title="교체 횟수", gridcolor="#F3F4F6"))
        st.plotly_chart(fig2, use_container_width=True)

        # 아웃카운트
        if "outs_before" in df.columns:
            st.markdown("#### 아웃카운트별 분포")
            outs_cnt = df.groupby("outs_before").size().reset_index(name="count")
            outs_cnt["label"] = outs_cnt["outs_before"].astype(str) + "아웃"
            fig3 = go.Figure(go.Bar(x=outs_cnt["label"], y=outs_cnt["count"],
                marker_color=LOTTE_NAVY, text=outs_cnt["count"], textposition="outside"))
            fig3.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=240,
                margin=dict(l=10,r=10,t=10,b=20),
                yaxis=dict(gridcolor="#F3F4F6"), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("점수 데이터 컬럼 없음")

with tab3:
    if has_wp and "lotte_wp_after" in df.columns:
        wp_df = df[df["lotte_wp_before"].notna() & df["lotte_wp_after"].notna()].copy()
        wp_df["wp_delta"] = wp_df["lotte_wp_after"] - wp_df["lotte_wp_before"]
        wp_df["direction"] = wp_df["wp_delta"].apply(lambda d: "개선" if d > 0.02 else ("악화" if d < -0.02 else "유지"))

        st.markdown("#### 교체 후 승리확률 변화 분포")
        fig4 = go.Figure(go.Histogram(
            x=wp_df["wp_delta"] * 100,
            nbinsx=40,
            marker_color=LOTTE_RED, opacity=0.75,
            name="WP 변화(%p)"
        ))
        fig4.add_vline(x=0, line_color=LOTTE_NAVY, line_width=2)
        fig4.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=280,
            margin=dict(l=10,r=10,t=20,b=20),
            xaxis=dict(title="WP 변화(%p)", showgrid=False),
            yaxis=dict(title="빈도", gridcolor="#F3F4F6"))
        st.plotly_chart(fig4, use_container_width=True)

        # 이닝별 WP 변화 평균
        st.markdown("#### 이닝별 평균 WP 변화")
        inning_wp = wp_df.groupby("inning")["wp_delta"].mean().reset_index()
        inning_wp = inning_wp[inning_wp["inning"] <= 12]
        fig5 = go.Figure(go.Bar(
            x=inning_wp["inning"], y=inning_wp["wp_delta"] * 100,
            marker_color=[LOTTE_RED if v > 0 else LOTTE_NAVY for v in inning_wp["wp_delta"]],
            text=[f"{v*100:+.1f}" for v in inning_wp["wp_delta"]], textposition="outside",
        ))
        fig5.add_hline(y=0, line_color="#9CA3AF", line_width=1)
        fig5.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=280,
            margin=dict(l=10,r=10,t=10,b=20),
            xaxis=dict(title="이닝", dtick=1, showgrid=False),
            yaxis=dict(title="평균 WP 변화(%p)", gridcolor="#F3F4F6"))
        st.plotly_chart(fig5, use_container_width=True)

        ca, cb, cc = st.columns(3)
        improved = (wp_df["direction"]=="개선").sum()
        worsened = (wp_df["direction"]=="악화").sum()
        neutral  = (wp_df["direction"]=="유지").sum()
        ca.metric("교체 후 개선", f"{improved}건", f"{improved/len(wp_df):.0%}")
        cb.metric("교체 후 악화", f"{worsened}건", f"{worsened/len(wp_df):.0%}")
        cc.metric("유지",         f"{neutral}건",  f"{neutral/len(wp_df):.0%}")
    else:
        st.info("PA 데이터에 lotte_wp_before / lotte_wp_after 컬럼이 없습니다. WhatIf 엔진으로 시뮬레이션 후 저장된 데이터가 있을 때 활성화됩니다.")

        # 대안: 교체 경기 vs 비교체 경기 승률
        if "lotte_win_label" in df.columns:
            st.markdown("#### 교체 횟수별 경기 승률 (대체 지표)")
            per_game = df.groupby("game_id").agg(
                교체수=("has_player_change","count"),
                승=("lotte_win_label","max"),
            ).reset_index()
            per_game["교체수_bin"] = pd.cut(per_game["교체수"], bins=[0,1,2,3,5,20],
                labels=["1회","2회","3회","4~5회","6회+"])
            grp = per_game.groupby("교체수_bin").agg(
                경기수=("game_id","count"), 승률=("승","mean")
            ).reset_index()
            fig6 = go.Figure(go.Bar(x=grp["교체수_bin"].astype(str), y=grp["승률"]*100,
                marker_color=LOTTE_RED, text=[f"{v:.1f}%" for v in grp["승률"]*100], textposition="outside"))
            fig6.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=260,
                margin=dict(l=10,r=10,t=10,b=20),
                yaxis=dict(title="승률(%)", range=[0,100], gridcolor="#F3F4F6"),
                xaxis=dict(title="경기당 투수 교체 횟수", showgrid=False))
            st.plotly_chart(fig6, use_container_width=True)

with tab4:
    st.markdown("#### 💡 데이터 기반 최적 타이밍 인사이트")

    inning_cnt2 = df.groupby("inning").size()
    peak_inning = int(inning_cnt2.idxmax())

    # 7회~ vs 6회이하 비율
    late_pct = (df["inning"] >= 7).mean()

    insights = [
        (f"가장 많이 교체하는 이닝: {peak_inning}회",
         f"전체 교체 중 {inning_cnt2.get(peak_inning,0):,}건이 {peak_inning}회에 집중. "
         f"이닝별 체력·피로도 관리를 이 시점 기준으로 준비해야 합니다."),
        (f"7회 이후 교체 비율: {late_pct:.0%}",
         "후반 불펜 운용 비중이 높을수록 마무리 투수 보존 전략이 중요합니다. "
         "7회 이전 교체를 줄이면 불펜 피로 누적을 낮출 수 있습니다."),
    ]

    if "batting_score_diff_before" in df.columns:
        losing_pct = (df["batting_score_diff_before"] < 0).mean()
        insights.append((
            f"뒤지는 상황에서 교체 비율: {losing_pct:.0%}",
            "실점 후 반응적 교체가 많을수록 선제적 교체 전략 도입이 WP를 높일 수 있습니다. "
            "동점·리드 상황에서 선제 교체를 늘리는 것이 효과적입니다."
        ))

    if "outs_before" in df.columns:
        early_out_pct = (df["outs_before"] == 0).mean()
        insights.append((
            f"0아웃 상황 교체 비율: {early_out_pct:.0%}",
            "이닝 시작 전 교체(0아웃)는 투구 수 절약에 유리하지만, "
            "상대 타순 재편성 리스크가 있습니다. 데이터 추적 후 세밀한 결정이 필요합니다."
        ))

    for title, body in insights:
        st.markdown(f'<div class="insight-card"><div class="insight-title">▸ {title}</div><div class="insight-body">{body}</div></div>', unsafe_allow_html=True)

    # 월별 교체 패턴
    if "month" in df.columns:
        st.markdown("#### 월별 교체 패턴")
        monthly = df.groupby("month").size().reset_index(name="교체횟수")
        if "lotte_win_label" in df.columns:
            monthly_wr = df.groupby("month")["lotte_win_label"].mean().reset_index()
            monthly_wr.columns = ["month","승률"]
            monthly = monthly.merge(monthly_wr, on="month")
            fig7 = go.Figure()
            fig7.add_trace(go.Bar(x=monthly["month"], y=monthly["교체횟수"],
                name="교체횟수", marker_color=LOTTE_NAVY, opacity=0.8))
            fig7.add_trace(go.Scatter(x=monthly["month"], y=monthly["승률"]*100,
                name="승률(%)", yaxis="y2", line=dict(color=LOTTE_RED, width=2.5),
                mode="lines+markers", marker=dict(size=8)))
            fig7.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=280,
                margin=dict(l=10,r=10,t=10,b=20),
                xaxis=dict(title="월", showgrid=False),
                yaxis=dict(title="교체 횟수", gridcolor="#F3F4F6"),
                yaxis2=dict(title="승률(%)", overlaying="y", side="right", showgrid=False, range=[0,100]),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig7, use_container_width=True)
        else:
            fig7 = go.Figure(go.Bar(x=monthly["month"], y=monthly["교체횟수"],
                marker_color=LOTTE_NAVY, text=monthly["교체횟수"], textposition="outside"))
            fig7.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=260,
                margin=dict(l=10,r=10,t=10,b=20),
                xaxis=dict(title="월", showgrid=False),
                yaxis=dict(title="교체 횟수", gridcolor="#F3F4F6"))
            st.plotly_chart(fig7, use_container_width=True)
