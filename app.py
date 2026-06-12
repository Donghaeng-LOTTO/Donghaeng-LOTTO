import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 글로벌 페이지 설정 및 다크 테마 테마링
# ==========================================
st.set_page_config(
    page_title="GIANTS AI Analytics", layout="wide", initial_sidebar_state="expanded"
)

# 가독성과 고대비(High Contrast)를 극대화한 핵심 CSS
st.markdown(
    """
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }

    .stApp {
        background: #090d16;
        color: #f1f5f9;
    }

    .dashboard-header {
        background: #111827;
        padding: 20px 30px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 25px;
    }
    .header-main-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #f8fafc;
    }
    .header-sub-title {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }

    .info-block {
        background: #111827;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        height: 100%;
    }
    .info-block-title {
        font-size: 14px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 12px;
        letter-spacing: 0.05em;
    }
    .info-block-content {
        font-size: 16px;
        font-weight: 500;
        color: #e2e8f0;
        line-height: 1.6;
    }

    .game-row {
        background: #111827;
        padding: 16px 24px;
        border-radius: 8px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        border: 1px solid #1f2937;
    }
    
    .player-card {
        background: #111827;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 15px;
    }
    .player-name {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
    }
    .player-pos {
        font-size: 13px;
        color: #3b82f6;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .verdict-box {
        padding: 24px;
        border-radius: 12px;
        margin-top: 25px;
        border: 1px solid transparent;
    }
    .verdict-box.should-have {
        background: rgba(239, 68, 68, 0.07);
        border-color: rgba(239, 68, 68, 0.3);
        border-left: 6px solid #ef4444;
    }
    .verdict-box.should-not {
        background: rgba(16, 185, 129, 0.07);
        border-color: rgba(16, 185, 129, 0.3);
        border-left: 6px solid #10b981;
    }
    .verdict-tag {
        font-size: 13px;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .verdict-tag.should-have { background: #ef4444; color: #ffffff; }
    .verdict-tag.should-not { background: #10b981; color: #ffffff; }
    
    .verdict-main-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .verdict-desc {
        font-size: 15px;
        color: #cbd5e1;
        line-height: 1.7;
    }

    div.stButton > button {
        background: #1f2937 !important;
        color: #94a3b8 !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Session State 시스템 매니저
# ==========================================
if "page" not in st.session_state:
    st.session_state["page"] = "main"
if "selected_game" not in st.session_state:
    st.session_state["selected_game"] = None


# ==========================================
# 3. 정적 가상 데이터 파이프라인
# ==========================================
def get_mock_schedule(year, month):
    return [
        {
            "id": 1,
            "date": f"{year}-{month:02d}-02",
            "matchup": "롯데 vs KIA",
            "result": "L (2:4)",
            "stadium": "광주",
            "situation": "9회말 2사 만루 1점차 상황. 타석에는 당일 무안타 기록 중인 9번 타자 배치. 상대 투수는 리그 최정상급 마무리.",
            "actual_action": "강공 유지 (기존 타자 대타 없이 출격)",
            "model_verdict": "should-have",
            "model_title": "대타 기용을 했어야 했다",
            "model_reason": "자체 연산 모델 분석 결과, 해당 상황에서 하위 타선의 강공 유지 시 득점 및 기대 승률은 14.2% 수준에 그쳤습니다. 반면 벤치 대기 중이던 타점 생산력 우수 베테랑 자원을 대타로 투입했을 시 기대 승률 시뮬레이션 최적값은 38.5%까지 도출되었습니다. 벤치의 강공 유지는 데이터 배치 관점에서 명백한 오판이었습니다.",
            "win_before": 14.2,
            "win_after": 38.5,
        },
        {
            "id": 2,
            "date": f"{year}-{month:02d}-05",
            "matchup": "롯데 vs 삼성",
            "result": "W (8:7)",
            "stadium": "사직",
            "situation": "7회초 1사 2, 3루 동점 위기 야기 상황. 볼카운트 투앤투에서 피안타율이 높은 몸쪽 커터 승부 유도.",
            "actual_action": "몸쪽 직구 정면 승부",
            "model_verdict": "should-not",
            "model_title": "몸쪽 직구 승부를 하지 말았어야 했다",
            "model_reason": "상대 타자의 당해 시즌 몸쪽 속구 카테고리 타율은 .345로 핫존에 해당합니다. 딥러닝 레이어가 제안한 최적 최저 피안타 코스는 외곽 슬라이더 떨어지는 궤적이었습니다. 결과적으로 실점은 면했으나 확률 기댓값 관점에서는 피안타 위험률이 62%에 달했던 도박성 짙은 로직이었습니다.",
            "win_before": 58.0,
            "win_after": 42.2,
        },
    ]


# ==========================================
# 4. 렌더링 파트 (들여쓰기 버그 완전 박멸)
# ==========================================
def render_header():
    st.markdown(
        """
<div class="dashboard-header">
    <div class="header-main-title">GIANTS AI ANALYTICS</div>
    <div class="header-sub-title">DEEP LEARNING BASEBALL DECISION SUPPORT SYSTEM</div>
</div>
""",
        unsafe_allow_html=True,
    )


def draw_win_probability_chart(before, after):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["실제 벤치 선택", "AI 최적 제안"],
            x=[before, after],
            orientation="h",
            marker=dict(color=["#374151", "#2563eb"]),
            text=[f"{before}%", f"{after}%"],
            textposition="auto",
            width=0.4,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8", "family": "Pretendard"},
        margin=dict(l=10, r=10, t=10, b=10),
        height=160,
        xaxis=dict(showgrid=False, range=[0, 100], visible=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=14, color="#ffffff")),
    )
    return fig


def render_main_dashboard():
    render_header()

    with st.sidebar:
        st.markdown(
            "<div style='padding:10px 0; font-weight:700; color:#64748b;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )
        menu_selection = st.radio(
            "Menu",
            ["경기 일정 분석", "선수 데이터베이스", "시즌 리더보드"],
            label_visibility="collapsed",
            key="sidebar_menu_radio",
        )

    # ------------------------------------------
    # 메뉴 1: 경기 일정 분석
    # ------------------------------------------
    if menu_selection == "경기 일정 분석":
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("시즌 선택", range(2023, 2027), index=3)
        with col2:
            month = st.selectbox("월 선택", range(3, 11), index=3)

        st.write("")
        games = get_mock_schedule(year, month)

        for game in games:
            col_info, col_btn = st.columns([6, 1])
            with col_info:
                st.markdown(
                    f"""
<div class="game-row">
    <div style="color: #64748b; font-size: 14px; width: 120px; font-weight:600;">{game['date']}</div>
    <div style="color: #f1f5f9; font-weight: 700; flex-grow: 1; font-size: 16px;">{game['matchup']}</div>
    <div style="color: #94a3b8; width: 100px; font-size: 14px;">{game['stadium']}</div>
    <div style="font-weight: 800; width: 80px; text-align: right; color: {'#10b981' if 'W' in game['result'] else '#ef4444'}">{game['result']}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with col_btn:
                st.write("")
                if st.button(
                    "분석 결과", key=f"btn_{game['id']}", use_container_width=True
                ):
                    st.session_state["page"] = "detail"
                    st.session_state["selected_game"] = game
                    st.rerun()

    # ------------------------------------------
    # 메뉴 2: 선수 데이터베이스 (들여쓰기 버그 수정 완료)
    # ------------------------------------------
    elif menu_selection == "선수 데이터베이스":
        st.markdown(
            "<h3 style='color:#ffffff; margin-bottom:20px;'>로스터 세부 데이터 분석</h3>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
<div class="player-card">
    <div class="player-name">황성빈</div>
    <div class="player-pos">외야수 (Outfielder)</div>
    <div style="color:#94a3b8; font-size:14px; margin-top:10px;">
        <b>시즌 타율:</b> .312<br>
        <b>출루율(OBP):</b> .385<br>
        <b>도루 성공률:</b> 88.5% (23/26)
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
<div class="player-card">
    <div class="player-name">레이예스</div>
    <div class="player-pos">외야수 (Outfielder)</div>
    <div style="color:#94a3b8; font-size:14px; margin-top:10px;">
        <b>시즌 타율:</b> .324<br>
        <b>장타율(SLG):</b> .492<br>
        <b>득점권 타율:</b> .356
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
<div class="player-card">
    <div class="player-name">박세웅</div>
    <div class="player-pos">선발투수 (Pitcher)</div>
    <div style="color:#94a3b8; font-size:14px; margin-top:10px;">
        <b>방어율(ERA):</b> 3.85<br>
        <b>WHIP:</b> 1.24<br>
        <b>QS 횟수:</b> 12회
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    # ------------------------------------------
    # 메뉴 3: 시즌 리더보드
    # ------------------------------------------
    elif menu_selection == "시즌 리더보드":
        st.markdown(
            "<h3 style='color:#ffffff; margin-bottom:20px;'>시즌 주요 지표 순위 현황</h3>",
            unsafe_allow_html=True,
        )

        leaderboard_data = pd.DataFrame(
            {
                "순위": [1, 2, 3, 4, 5],
                "팀명": ["KIA", "삼성", "LG", "두산", "롯데"],
                "경기수": [144, 144, 144, 144, 144],
                "승": [87, 82, 78, 74, 72],
                "패": [55, 60, 64, 68, 70],
                "무": [2, 2, 2, 2, 2],
                "승률": [0.613, 0.577, 0.549, 0.521, 0.507],
            }
        )
        st.dataframe(leaderboard_data.set_index("순위"), use_container_width=True)


def render_detail_page():
    game = st.session_state["selected_game"]

    if st.button("← 전체 일정 목록으로 이동"):
        st.session_state["page"] = "main"
        st.session_state["selected_game"] = None
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div style="background: #111827; padding: 24px; border-radius: 12px; border: 1px solid #1f2937; margin-bottom: 20px;">
    <div style="color: #3b82f6; font-weight: 700; font-size: 12px; letter-spacing:0.05em;">MATCH ANALYSIS</div>
    <div style="font-size: 26px; font-weight: 800; color: #ffffff; margin-top:4px;">{game['matchup']}</div>
    <div style="color: #64748b; font-size: 14px; margin-top: 6px;">{game['date']} | {game['stadium']} | 경기 결과: {game['result']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(
            f"""
<div class="info-block">
<div class="info-block-title">경기 핵심 분기점 (Turning Point)</div>
<div class="info-block-content">{game['situation']}</div>
<div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #1f2937;">
<div style="font-size: 13px; color: #64748b; font-weight:700; margin-bottom:4px;">실제 경기 당시 벤치의 선택</div>
<div style="font-size: 18px; font-weight: 700; color: #ffffff;">{game['actual_action']}</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="info-block">', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-block-title">의사결정별 기대 승률 변동 기댓값</div>',
            unsafe_allow_html=True,
        )
        fig = draw_win_probability_chart(game["win_before"], game["win_after"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    verdict_class = game["model_verdict"]
    badge_text = "RECOMMENDED" if verdict_class == "should-have" else "CRITICAL ERROR"

    st.markdown(
        f"""
<div class="verdict-box {verdict_class}">
<div class="verdict-tag {verdict_class}">{badge_text}</div>
<div class="verdict-main-title">{game['model_title']}</div>
<div style="font-size: 13px; color: #64748b; font-weight: 600; margin-bottom: 12px;">Deep Learning Model Diagnostic Log</div>
<p class="verdict-desc">{game['model_reason']}</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================
# 5. 라우터 컨트롤러
# ==========================================
if st.session_state["page"] == "main":
    render_main_dashboard()
elif st.session_state["page"] == "detail":
    render_detail_page()
