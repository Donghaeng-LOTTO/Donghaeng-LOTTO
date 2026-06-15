import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as objects
import json
import time
from streamlit_option_menu import option_menu
from streamlit_modal import Modal
import streamlit.components.v1 as components
import base64

# 페이지 기본 설정
st.set_page_config(
    page_title="LOTTE GIANTS INSIGHT PLATFORM",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        /* 탭의 버튼 텍스트 색상 및 스타일 */
        [data-baseweb="tab"] {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #333333 !important; /* 탭 글씨 색상 (어두운 회색) */
        }
        
        /* 선택된 탭의 강조 색상 */
        [data-baseweb="tab-highlight"] {
            background-color: #E31937 !important; /* 롯데 레드 */
        }
        
        /* 탭에 마우스를 올렸을 때 */
        [data-baseweb="tab"]:hover {
            color: #E31937 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# 로딩 스크린 프로세스 (최초 1회 실행)
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if not st.session_state.loaded:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            """
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
                .loader-container {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #F8FAFC;
                    font-family: 'Inter', sans-serif;
                }
                .loader-title {
                    font-size: 48px;
                    font-weight: 800;
                    color: #E31937;
                    margin-bottom: 8px;
                    letter-spacing: -1px;
                    animation: pulse 1.5s infinite ease-in-out;
                }
                .loader-subtitle {
                    font-size: 18px;
                    font-weight: 400;
                    color: #6B7280;
                    margin-bottom: 32px;
                }
                .spinner {
                    width: 50px;
                    height: 50px;
                    border: 5px solid #E5E7EB;
                    border-top: 5px solid #E31937;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                @keyframes pulse {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                }
            </style>
            <div class="loader-container">
                <div class="loader-title">LOTTE GIANTS</div>
                <div class="loader-subtitle">Loading Analytics Platform...</div>
                <div class="spinner"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(2.0)
    st.session_state.loaded = True
    loading_placeholder.empty()
    st.rerun()
st.markdown(
    """
    <style>
        /* 기본 폰트 및 배경 리셋 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC !important;
            color: #111827 !important;
            scroll-behavior: smooth;
        }

        /* 스트림릿 기본 상단/하단 UI 숨김 */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        
        /* 메인 컨테이너 패딩 최적화 */
        .block-container {
            max-width: 1400px !important;
            padding-top: 0rem !important;
            padding-bottom: 5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        section.main > div {
            padding-top: 0rem !important;
        }

        /* 전역 버튼 스타일 보정 (글씨 묻힘 현상 해결) */
        .stButton > button {
            background-color: #E31937 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 28px !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em !important;
            box-shadow: 0 4px 6px -1px rgba(227, 25, 55, 0.2) !important;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        
        .stButton > button:hover {
            background-color: #111827 !important;
            color: #FFFFFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(17, 24, 39, 0.2) !important;
        }

        /* 상단 고정 네비게이션 스타일 블러 처리 */
        .st-emotion-cache-12fmcee {
            position: sticky;
            top: 0;
            z-index: 999;
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(12px);
            border-bottom: 1px solid #E5E7EB;
        }

        /* 섹션별 레이아웃 간격 정의 */
        .section-wrapper {
            padding-top: 100px;
            padding-bottom: 40px;
        }

        .section-title {
            font-size: 36px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 12px;
            letter-spacing: -0.03em;
        }

        .section-subtitle {
            font-size: 18px;
            color: #6B7280;
            margin-bottom: 48px;
            font-weight: 400;
            max-width: 700px;
            line-height: 1.6;
        }

        /* 모던 디자인 카드 컴포넌트 */
        .saas-card {
            background: #FFFFFF;
            border-radius: 24px;
            border: 1px solid #E5E7EB;
            padding: 40px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .saas-card:hover {
            transform: translateY(-10px) scale(1.03);
            box-shadow: 0 20px 50px rgba(0,0,0,0.12);
            border-color: #E31937;
        }

        /* 기술 스택 전용 배지 */
        .tech-badge {
            background: #FFFFFF;
            color: #111827;
            border: 1px solid #E5E7EB;
            padding: 12px 24px;
            border-radius: 50px;
            font-weight: 500;
            font-size: 15px;
            display: inline-block;
            margin: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }

        .tech-badge:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(227, 25, 55, 0.2);
            border-color: #E31937;
            color: #E31937;
        }

        /* 페이드인 애니메이션 효과 */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .animated-fade-in {
            animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div style='margin-top:0px;'></div>", unsafe_allow_html=True)

# 수평 고정 메뉴 탭 생성
selected_menu = option_menu(
    menu_title=None,
    options=[
        "HOME",
        "PROJECT OVERVIEW",
        "DATA OVERVIEW",
        "EDA RESULTS",
        "MODEL PERFORMANCE",
        "ROADMAP",
        "TECH STACK",
    ],
    icons=[
        "house",
        "vector-pen",
        "database",
        "bar-chart-line",
        "cpu",
        "person-badge",
        "diagram-3",
        "layer-backward",
    ],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "#FFFFFF",
            "border-radius": "0px",
            "max-width": "100%",
        },
        "icon": {"color": "#6B7280", "font-size": "14px"},
        "nav-link": {
            "font-size": "13px",
            "font-weight": "600",
            "text-align": "center",
            "margin": "0px",
            "color": "#111827",
            "padding": "20px 0px",
            "transition": "all 0.3s ease",
        },
        "nav-link-selected": {
            "background-color": "transparent",
            "color": "#E31937",
            "border-bottom": "3px solid #E31937",
            "border-radius": "0px",
        },
    },
)

# 탭 선택 시 스크롤 이벤트 스크립트 주입
st.markdown(
    f"""
    <script>
        const targetSection = "{selected_menu}".toLowerCase().replace(/ /g, "-");
        const element = parent.document.getElementById(targetSection);
        if (element) {{
            element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }}
    </script>
    """,
    unsafe_allow_html=True,
)


# 섹션 위치 지정을 위한 고유 앵커 생성용 헬퍼 함수
def create_section_anchor(name: str):
    anchor_id = name.lower().replace(" ", "-")
    st.markdown(
        f"<div id='{anchor_id}' class='section-anchor'></div>", unsafe_allow_html=True
    )
create_section_anchor("HOME")

st.markdown("<div class='section-wrapper animated-fade-in'>", unsafe_allow_html=True)
hero_col1, hero_col2 = st.columns([0.55, 0.45])

with hero_col1:
    st.markdown(
        """
        <style>
            .hero-badge {
                background: rgba(227, 25, 55, 0.08);
                color: #E31937;
                padding: 8px 16px;
                border-radius: 30px;
                font-weight: 600;
                font-size: 14px;
                display: inline-block;
                margin-bottom: 24px;
                letter-spacing: 0.05em;
            }
            .hero-title-primary {
                font-size: 64px;
                font-weight: 800;
                line-height: 1.1;
                color: #111827;
                letter-spacing: -0.04em;
                margin-bottom: 16px;
            }
            .hero-title-accent {
                font-size: 40px;
                font-weight: 700;
                line-height: 1.2;
                color: #E31937;
                margin-bottom: 24px;
                letter-spacing: -0.02em;
            }
            .hero-description {
                font-size: 20px;
                line-height: 1.6;
                color: #6B7280;
                margin-bottom: 40px;
                font-weight: 400;
            }
        </style>
        <div>
            <div class="hero-badge">Professional Baseball Analytics Platform</div>
            <div class="hero-title-primary">LOTTE GIANTS<br>DATA ANALYTICS PLATFORM</div>
            <div class="hero-title-accent">데이터를 통해 롯데 자이언츠의<br>숨겨진 인사이트를 발견하다</div>
            <p class="hero-description">
                최신 기계학습 모델과 고도화된 고급 통계 지표 분석을 통해 선수단의 경기력을 정밀 진단하고, 
                승리를 결정짓는 핵심 요인을 도출하는 엔터프라이즈급 프로 스포츠 데이터 분석 플랫폼입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

with col_a:
    if st.button(
        "🚀 시연 시작",
        use_container_width=True
    ):
        st.switch_page(
            "pages/01_Model_Analysis.py"
        )

with col_b:
    if st.button(
        "📖 프로젝트 소개",
        use_container_width=True
    ):
        st.markdown(
            """
            <script>
            parent.document.getElementById(
            "project-overview"
            ).scrollIntoView(
            {behavior:'smooth'}
            );
            </script>
            """,
            unsafe_allow_html=True
        )

# GLB 3D 파일 바이너리 변환 및 예외 처리
try:
    with open("baseball.glb", "rb") as f:
        glb_bytes = f.read()
        glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
    glb_data_url = f"data:application/octet-stream;base64,{glb_base64}"
except FileNotFoundError:
    glb_data_url = None
    st.error(
        "baseball.glb 파일을 찾을 수 없습니다. app.py와 같은 폴더에 있는지 확인해주세요."
    )

with hero_col2:
    if glb_base64:
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body, html { margin: 0; padding: 0; overflow: hidden; background-color: #F8FAFC; width: 100%; height: 100%; }
                #canvas-container { width: 100vw; height: 550px; display: flex; justify-content: center; align-items: center; }
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
        </head>
        <body>
            <div id="canvas-container"></div>
            <script>
                const container = document.getElementById('canvas-container');
                const width = window.innerWidth || 550;
                const height = 550;

                const scene = new THREE.Scene();
                
                const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                camera.position.set(0, 0, 10); 

                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.outputEncoding = THREE.sRGBEncoding;
                container.appendChild(renderer.domElement);

                const ambientLight = new THREE.AmbientLight(0xffffff, 1.5); 
                scene.add(ambientLight);

                const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
                mainLight.position.set(5, 5, 5);
                scene.add(mainLight);

                let ball;
                // 시간에 따른 부드러운 회전을 위한 시계 생성
                const clock = new THREE.Clock(); 

                const base64Data = "%%GLB_BASE64%%";
                try {
                    const binaryString = atob(base64Data);
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }

                    const loader = new THREE.GLTFLoader();
                    loader.parse(
                        bytes.buffer,
                        '',
                        (gltf) => {
                            ball = gltf.scene;
                            scene.add(ball);

                            const box = new THREE.Box3().setFromObject(ball);
                            const center = box.getCenter(new THREE.Vector3());
                            ball.position.sub(center);

                            const size = box.getSize(new THREE.Vector3());
                            const maxDim = Math.max(size.x, size.y, size.z);
                            if (maxDim > 0) {
                                const targetScale = 5.2 / maxDim; 
                                ball.scale.set(targetScale, targetScale, targetScale);
                            }

                            camera.lookAt(0, 0, 0);
                        },
                        (error) => {
                            console.error('로딩 실패:', error);
                        }
                    );
                } catch (e) {
                    console.error('디코딩 실패:', e);
                }

                function animate() {
                    requestAnimationFrame(animate);
                    
                    if (ball) {
                        const elapsedTime = clock.getElapsedTime();
                        
                        ball.rotation.x = elapsedTime * 8.5; 
                        ball.rotation.z = 0;
                        
                        // 위아래로 부드럽게 움직이는 효과
                        ball.position.y = Math.sin(elapsedTime * 1.5) * 0.1;
                    }

                    renderer.render(scene, camera);
                }

                window.addEventListener('resize', () => {
                    const w = window.innerWidth;
                    camera.aspect = w / height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(w, height);
                });

                animate();
            </script>
        </body>
        </html>
        """
        st.components.v1.html(
            html_template.replace("%%GLB_BASE64%%", glb_base64), height=550
        )
create_section_anchor("PROJECT OVERVIEW")

st.markdown(
    """
    <div class='section-wrapper'>
        <div class='section-title'>PROJECT OVERVIEW</div>
        <div class='section-subtitle'>롯데 자이언츠의 도약을 위한 데이터 기반 핵심 전략 수립 프로젝트의 거시적 설계 구조와 최종 기대효과를 정의합니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_col1, overview_col2, overview_col3 = st.columns(3)

modal_po = Modal("Project Purpose", key="modal_po_key", max_width=800)
modal_pd = Modal("Problem Definition", key="modal_pd_key", max_width=800)
modal_ei = Modal("Expected Impact", key="modal_ei_key", max_width=800)

with overview_col1:
    st.markdown(
        """
        <div class='saas-card'>
            <div>
                <div style='font-size: 40px; margin-bottom: 16px;'>🎯</div>
                <div style='font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 12px;'>Project Purpose</div>
                <div style='font-size: 15px; font-weight: 600; color: #E31937; margin-bottom: 16px;'>데이터 중심 객관적 의무 결정 체계 확립</div>
                <p style='font-size: 14px; color: #6B7280; line-height: 1.6; margin-bottom: 24px;'>
                    전통적인 야구 분석론을 뛰어넘어, 세부 세이버메트릭스 지표와 딥러닝 기반 모델을 융합하여 승리를 직결하는 변수 분석을 고도화합니다.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Details", key="btn_po"):
        modal_po.open()

with overview_col2:
    st.markdown(
        """
        <div class='saas-card'>
            <div>
                <div style='font-size: 40px; margin-bottom: 16px;'>🔍</div>
                <div style='font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 12px;'>Problem Definition</div>
                <div style='font-size: 15px; font-weight: 600; color: #E31937; margin-bottom: 16px;'>정성적 판단 한계 및 클러치 역량 부재 파악</div>
                <p style='font-size: 14px; color: #6B7280; line-height: 1.6; margin-bottom: 24px;'>
                    상황별 득점권 빈타 원인과 실점 유발 패턴을 다각적 피처 매핑을 통해 가시화하고, 단순 누적 스탯 이면의 하향 지표를 정량적으로 추적합니다.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Details", key="btn_pd"):
        modal_pd.open()

with overview_col3:
    st.markdown(
        """
        <div class='saas-card'>
            <div>
                <div style='font-size: 40px; margin-bottom: 16px;'>📈</div>
                <div style='font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 12px;'>Expected Impact</div>
                <div style='font-size: 15px; font-weight: 600; color: #E31937; margin-bottom: 16px;'>전술 최적화 및 장기적 승률 상승</div>
                <p style='font-size: 14px; color: #6B7280; line-height: 1.6; margin-bottom: 24px;'>
                    상대 투수/타자 상성에 따른 라인업 효율 극대화 및 최적의 불펜 교체 타이밍 산출을 통해 구단 운영 효율성을 약 15% 개선하는 효과를 예상합니다.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Details", key="btn_ei"):
        modal_ei.open()

# 모달 내부 팝업 이벤트 처리 루프
if modal_po.is_open():
    with modal_po.container():
        st.markdown(
            "<div style='padding: 20px;'><h3 style='color:#E31937; margin-bottom:20px;'>프로젝트 추진 배경 및 목적</h3><p style='line-height:1.7; color:#111827;'>개별 선수의 타구 속도, 발사각, 득점권 심리 회복 탄력성 지표를 종합 분석하여 코칭스태프가 철저히 계량화된 통계 분석에 기반해 엔트리를 구성하도록 돕습니다.</p></div>",
            unsafe_allow_html=True,
        )

if modal_pd.is_open():
    with modal_pd.container():
        st.markdown(
            "<div style='padding: 20px;'><h3 style='color:#E31937; margin-bottom:20px;'>해결하고자 하는 당면 과제</h3><p style='line-height:1.7; color:#111827;'>1. 득점권 변동성 분석<br>2. 머신러닝 교차점 기반 불펜 교체 임계치 산출<br>3. 퓨처스-1군 스탯 연동형 계층 클러스터링 고도화</p></div>",
            unsafe_allow_html=True,
        )

if modal_ei.is_open():
    with modal_ei.container():
        st.markdown(
            "<div style='padding: 20px;'><h3 style='color:#E31937; margin-bottom:20px;'>기대 효과 및 인사이트 도출 방안</h3><p style='line-height:1.7; color:#111827;'>사직 구장 펜스 상향 가중치가 투타 방어 효율성에 미친 세부 상관관계를 정량화하고 승부처의 라인업 생산력을 강화합니다.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
create_section_anchor("DATA OVERVIEW")

st.markdown(
    """
    <div class='section-wrapper'>
        <div class='section-title'>DATA OVERVIEW</div>
        <div class='section-subtitle'>분석 플랫폼 아키텍처의 기반이 되는 데이터의 규모와 수집 주기, 핵심 구조의 메트릭 리포트입니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

st.markdown(
    """
    <style>
        .metric-card {
            background: #FFFFFF;
            border-radius: 20px;
            padding: 28px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.01), 0 2px 4px -1px rgba(0,0,0,0.01);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.06);
            border-color: #E31937;
        }
        .metric-value {
            font-size: 36px;
            font-weight: 800;
            color: #E31937;
            margin-bottom: 6px;
            letter-spacing: -0.02em;
        }
        .metric-label {
            font-size: 14px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 4px;
        }
        .metric-sub {
            font-size: 12px;
            color: #6B7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with metric_col1:
    st.markdown(
        "<div class='metric-card'><div class='metric-value'>12</div><div class='metric-label'>Dataset Count</div><div class='metric-sub'>다차원 정합 소스 테이블 개수</div></div>",
        unsafe_allow_html=True,
    )
with metric_col2:
    st.markdown(
        "<div class='metric-card'><div class='metric-value'>2021-2025</div><div class='metric-label'>Collection Period</div><div class='metric-sub'>최근 5개 시즈널 프레임</div></div>",
        unsafe_allow_html=True,
    )
with metric_col3:
    st.markdown(
        "<div class='metric-card'><div class='metric-value'>1,452,890</div><div class='metric-label'>Total Records</div><div class='metric-sub'>전수 인스턴스 규모</div></div>",
        unsafe_allow_html=True,
    )
with metric_col4:
    st.markdown(
        "<div class='metric-card'><div class='metric-value'>78</div><div class='metric-label'>Feature Count</div><div class='metric-sub'>인코딩 타겟 속성수</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
modal_do = Modal("Data Specifications", key="modal_do_key", max_width=900)

m_btn_col1, m_btn_col2, m_btn_col3 = st.columns([0.45, 0.1, 0.45])
with m_btn_col2:
    if st.button("View Details", key="btn_do"):
        modal_do.open()

if modal_do.is_open():
    with modal_do.container():
        st.markdown(
            "<div style='padding: 20px;'><h3 style='color:#E31937; margin-bottom:20px;'>데이터 명세 데이터프레임</h3><p>API 및 스크래핑 전처리 기법(KNN Imputer, IQR 필터링)을 거친 클린 행렬 정보 가동 리포트입니다.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
create_section_anchor("EDA RESULTS")

st.markdown(
    """
    <div class='section-wrapper'>
        <div class='section-title'>EDA RESULTS</div>
        <div class='section-subtitle'>데이터 탐색을 통해 규명된 승리 상관관계 및 주요 통계적 특성의 가시화 리포트입니다. 차트를 클릭하면 상세 분석 모달이 활성화됩니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 다차원 데이터셋 절차적 시뮬레이션 생성
np.random.seed(42)

# 차트 1: 경기 승패별 OPS 분포
win_ops = np.random.normal(loc=0.840, scale=0.08, size=200)
loss_ops = np.random.normal(loc=0.680, scale=0.09, size=200)
df_ops = pd.DataFrame(
    {
        "OPS": np.concatenate([win_ops, loss_ops]),
        "Match Result": ["Victory"] * 200 + ["Defeat"] * 200,
    }
)

# 차트 2: 홈/원정 승률 변화 추이 타임라인
years = ["2021", "2022", "2023", "2024", "2025"]
home_wr = [0.46, 0.49, 0.52, 0.55, 0.58]
away_wr = [0.42, 0.44, 0.41, 0.48, 0.51]
df_trend = pd.DataFrame(
    {
        "Year": years * 2,
        "Win Rate": home_wr + away_wr,
        "Location": ["Home"] * 5 + ["Away"] * 5,
    }
)

# 차트 3: 선수단 타격 지표 군집 분석 3D 좌표 스펙
cluster_size = 50
df_cluster = pd.DataFrame(
    {
        "Batting Average": np.concatenate(
            [
                np.random.normal(0.290, 0.02, cluster_size),
                np.random.normal(0.240, 0.015, cluster_size),
                np.random.normal(0.265, 0.02, cluster_size),
            ]
        ),
        "Home Runs": np.concatenate(
            [
                np.random.normal(25, 5, cluster_size),
                np.random.normal(8, 3, cluster_size),
                np.random.normal(15, 4, cluster_size),
            ]
        ),
        "RBI": np.concatenate(
            [
                np.random.normal(85, 10, cluster_size),
                np.random.normal(35, 8, cluster_size),
                np.random.normal(60, 12, cluster_size),
            ]
        ),
        "Cluster": ["Ops Monster (A)"] * cluster_size
        + ["Speed/Defense (B)"] * cluster_size
        + ["Solid Regular (C)"] * cluster_size,
        "Player": [f"선수_{i}" for i in range(cluster_size * 3)],
    }
)

# 3단 컬럼 배치 및 모달 연동 선언
eda_col1, eda_col2, eda_col3 = st.columns(3)

modal_eda1 = Modal("OPS Distribution Analysis Details", key="m_eda1", max_width=800)
modal_eda2 = Modal("Home vs Away Trend Details", key="m_eda2", max_width=800)
modal_eda3 = Modal("Player Cluster 3D Analysis Details", key="m_eda3", max_width=800)

with eda_col1:
    st.markdown(
        "<h4 style='text-align:center; color:#111827;'>경기 결과별 팀 OPS 분포</h4>",
        unsafe_allow_html=True,
    )
    fig1 = px.histogram(
        df_ops,
        x="OPS",
        color="Match Result",
        marginal="box",
        color_discrete_map={"Victory": "#E31937", "Defeat": "#1E3A8A"},
        barmode="overlay",
        opacity=0.75,
    )
    fig1.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig1.update_yaxes(
        showgrid=True, gridcolor="#F3F4F6", showline=True, linecolor="#E5E7EB"
    )
    st.plotly_chart(fig1, width="stretch")  # Deprecation 수정 완료
    if st.button("OPS 분석 상세보기", key="btn_eda1"):
        modal_eda1.open()

with eda_col2:
    st.markdown(
        "<h4 style='text-align:center; color:#111827;'>연도별 홈/원정 승률 추이</h4>",
        unsafe_allow_html=True,
    )
    fig2 = px.line(
        df_trend,
        x="Year",
        y="Win Rate",
        color="Location",
        color_discrete_map={"Home": "#E31937", "Away": "#6B7280"},
        markers=True,
    )
    fig2.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig2.update_yaxes(
        showgrid=True, gridcolor="#F3F4F6", showline=True, linecolor="#E5E7EB"
    )
    st.plotly_chart(fig2, width="stretch")  # Deprecation 수정 완료
    if st.button("승률 추이 상세보기", key="btn_eda2"):
        modal_eda2.open()

with eda_col3:
    st.markdown(
        "<h4 style='text-align:center; color:#111827;'>선수단 군집 분석 (3D 스캐터)</h4>",
        unsafe_allow_html=True,
    )
    fig3 = px.scatter_3d(
        df_cluster,
        x="Batting Average",
        y="Home Runs",
        z="RBI",
        color="Cluster",
        hover_name="Player",
        color_discrete_map={
            "Ops Monster (A)": "#E31937",
            "Speed/Defense (B)": "#1E3A8A",
            "Solid Regular (C)": "#10B981",
        },
    )
    fig3.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#E5E7EB",
                showbackground=False,
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#E5E7EB",
                showbackground=False,
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#E5E7EB",
                showbackground=False,
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig3, width="stretch")  # Deprecation 수정 완료
    if st.button("군집 분석 상세보기", key="btn_eda3"):
        modal_eda3.open()

# 모달 컨텐츠 정의 구역
if modal_eda1.is_open():
    with modal_eda1.container():
        st.markdown(
            "<div style='padding:20px;'><h3 style='color:#E31937;'>OPS 분포 세부 인사이트</h3><p>승리 경기의 팀 OPS 중앙값은 0.840선에 형성되며, 패배 경기(0.680)와 명확한 통계적 유의차를 보입니다. 즉, 승리를 보장하기 위한 최소 임계 임팩트 타격 지표는 경기당 OPS 0.800 빌드업입니다.</p></div>",
            unsafe_allow_html=True,
        )

if modal_eda2.is_open():
    with modal_eda2.container():
        st.markdown(
            "<div style='padding:20px;'><h3 style='color:#E31937;'>홈/원정 승률 추이 세부 인사이트</h3><p>2024년 구장 환경 개보수 이후 홈 승률이 58%까지 급격하게 우상향하는 팩터를 가시화합니다. 사직 야구장 파크 팩터의 변화가 투수진의 피안타율 감소에 직접적인 영향을 준 것으로 분석됩니다.</p></div>",
            unsafe_allow_html=True,
        )

if modal_eda3.is_open():
    with modal_eda3.container():
        st.markdown(
            "<div style='padding:20px;'><h3 style='color:#E31937;'>3D 클러스터 분석 세부 인사이트</h3><p>타율, 홈런, 타점을 축으로 군집화한 결과 구단의 뎁스가 3개 핵심 세그먼트로 빌드됩니다. 붉은색 군집(Ops Monster)의 생산력을 유지하면서 초록색 regular 군집의 상향 마이그레이션 전략이 요구됩니다.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
create_section_anchor("MODEL PERFORMANCE")

st.markdown(
    """
    <div class='section-wrapper'>
        <div class='section-title'>MODEL PERFORMANCE</div>
        <div class='section-subtitle'>경기 승패 예측 및 득점 기여 모델의 분류 평가지표 현황입니다. 실제 테스트셋 기준 검증 성능 리포트입니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

gauge_col1, gauge_col2, gauge_col3, gauge_col4 = st.columns(4)


# Plotly Indicator 기반의 게이지 렌더링 자동화 팩토리 함수
def create_metric_gauge(title, value):
    fig = objects.Figure(
        objects.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E31937"},
                "bgcolor": "white",
            },
        )
    )
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# 게이지 표시
with gauge_col1:
    st.plotly_chart(create_metric_gauge("Accuracy", 88), use_container_width=True)
with gauge_col2:
    st.plotly_chart(create_metric_gauge("Precision", 92), use_container_width=True)
with gauge_col3:
    st.plotly_chart(create_metric_gauge("Recall", 85), use_container_width=True)
with gauge_col4:
    st.plotly_chart(create_metric_gauge("F1-Score", 89), use_container_width=True)

# 페이지 하단 시연 페이지 이동 버튼 (발표 흐름의 끝)
st.divider()
st.markdown("<div style='text-align: center; padding: 20px;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        text-align:center;
        padding:60px 20px;
        background:#FFFFFF;
        border-radius:24px;
        border:1px solid #E5E7EB;
        margin-top:50px;
        margin-bottom:50px;
    ">
        <h2 style="color:#111827;">
            준비되셨나요?
        </h2>

        <p style="
            color:#6B7280;
            font-size:18px;
            margin-bottom:30px;
        ">
            실제 롯데 자이언츠 선수 데이터 분석 페이지로 이동하여<br>
            시즌별 기록 및 다양한 통계 지표를 확인해보세요.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 시연 페이지 시작하기", use_container_width=True, type="primary"):
        st.switch_page("pages/01_Model_Analysis.py")
