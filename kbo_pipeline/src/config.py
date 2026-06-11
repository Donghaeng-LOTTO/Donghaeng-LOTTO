"""전역 설정.

모든 경로/상수/팀코드/시즌 범위를 한 곳에서 관리한다.
"""
from pathlib import Path

# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = Path("data")

RAW_DIR = BASE_DIR / "raw"
RAW_SCHEDULE_DIR = RAW_DIR / "naver_schedule"
RAW_RECORD_DIR = RAW_DIR / "naver_record"
RAW_RELAY_DIR = RAW_DIR / "naver_relay"

PROCESSED_DIR = BASE_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"

KAGGLE_DIR = BASE_DIR / "external" / "kaggle"
KAGGLE_BATTING_BY_SEASON = KAGGLE_DIR / "kbo_batting_stats_by_season_1982-2025.csv"
KAGGLE_PITCHING_BY_SEASON = KAGGLE_DIR / "kbo_pitching_stats_by_season_1982-2025.csv"

FAILED_REQUESTS_CSV = LOG_DIR / "failed_requests.csv"

ALL_DIRS = [
    RAW_SCHEDULE_DIR,
    RAW_RECORD_DIR,
    RAW_RELAY_DIR,
    PROCESSED_DIR,
    LOG_DIR,
]

# ------------------------------------------------------------
# 네이버 API
# ------------------------------------------------------------
SCHEDULE_URL = "https://api-gw.sports.naver.com/schedule/games"
RECORD_URL = "https://api-gw.sports.naver.com/schedule/games/{game_id}/record"
RELAY_URL = "https://api-gw.sports.naver.com/schedule/games/{game_id}/relay"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "origin": "https://m.sports.naver.com",
    "referer": "https://m.sports.naver.com/",
    "user-agent": "Mozilla/5.0",
    "x-sports-backend": "kotlin",
}

# 요청 간 대기(초). 대량 수집 시 0.3~0.5 권장.
DEFAULT_SLEEP_SEC = 0.4
# 실패 시 재시도
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0
REQUEST_TIMEOUT = 10

# ------------------------------------------------------------
# KBO 팀 코드
# ------------------------------------------------------------
REAL_KBO_TEAM_CODES = {
    "LG", "OB", "HT", "HH", "LT", "SS", "SK", "WO",
    "KI", "NC", "KT", "SSG", "KW", "HE",
}

# record fallback에서 game_id 후보 생성용. 연도별 실제 참가 팀 코드.
def team_codes_for_year(year: int) -> list[str]:
    codes = ["LG", "OB", "SS", "HH", "HT", "LT"]
    # SK -> SSG (2021~)  *네이버 game_id 상 코드는 SK 유지인 경우가 많아 둘 다 후보에 넣는다
    codes.append("SK")
    if year >= 2021:
        codes.append("SSG")
    # 우리/넥센/키움: WO 코드 유지
    if year >= 2008:
        codes.append("WO")
    if year >= 2013:
        codes.append("NC")
    if year >= 2015:
        codes.append("KT")
    return codes


# ------------------------------------------------------------
# 시즌 수집 범위 (월 단위)
# KBO 정규시즌+포스트시즌을 넉넉히 커버: 3월 1일 ~ 11월 30일
# ------------------------------------------------------------
SEASON_START_MMDD = "0301"
SEASON_END_MMDD = "1130"

LOTTE_TEAM_CODE = "LT"
