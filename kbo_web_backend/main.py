# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(title="KBO What-if API")

# React(3000포트) 연동을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 바로 옆 kbo_pipeline 폴더의 산출물 위치 설정
DATA_DIR = "../kbo_pipeline/data/processed"
PLAYERS_CSV = os.path.join(DATA_DIR, "naver_players_seen.csv")


@app.get("/")
def read_root():
    return {"message": "kbo_web_backend 서버가 정상 작동 중입니다!"}


@app.get("/api/players")
def get_players():
    """kbo_pipeline의 결과물인 선수 마스터 CSV를 읽어와 반환"""
    if not os.path.exists(PLAYERS_CSV):
        return [
            {"pcode": "79105", "name": "류현진(테스트용)"},
            {"pcode": "54001", "name": "김도영(테스트용)"},
        ]

    # CSV 읽기 및 중복 제거 후 React 전달용 구조 변환
    df = pd.read_csv(PLAYERS_CSV)
    players_data = (
        df[["pcode", "name"]].drop_duplicates().head(100).to_dict(orient="records")
    )
    return players_data
