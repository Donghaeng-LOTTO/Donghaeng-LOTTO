"""games.csv 수집.

1차: schedule API
2차: schedule 실패/유효 경기 0개 -> record fallback

핵심:
- schedule 응답 구조가 바뀌어도 gameId가 있는 dict를 재귀적으로 찾음
- 구버전 키(aCode/hCode)와 신버전 키(awayTeamCode/homeTeamCode)를 모두 지원
- game_id 앞 8자리 == 요청 날짜 검증
- 비정규 항목(BMBC, CHITCHIT, SPORTSN 등) 제외
- games.csv가 비어 있어도 pandas EmptyDataError가 나지 않게 처리
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src import config
from src.clients.naver_api import (
    NaverClient,
    ensure_dirs,
    log_failed_request,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# games.csv 컬럼 정의
# ------------------------------------------------------------
GAME_COLUMNS = [
    "game_id",
    "game_date",
    "game_time",
    "weekday",
    "away_team_code",
    "away_team_name",
    "away_team_full_name",
    "home_team_code",
    "home_team_name",
    "home_team_full_name",
    "stadium",
    "status_code",
    "status_info",
    "cancel_flag",
    "suspended_flag",
    "dheader",
    "away_score",
    "home_score",
]


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
def daterange(start_date: str, end_date: str):
    """YYYYMMDD 문자열 범위를 하루씩 순회."""
    cur = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def pick(obj: dict, *keys: str, default=None):
    """여러 후보 키 중 처음으로 값이 존재하는 것을 반환."""
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return default


def find_game_objects(obj: Any) -> list[dict]:
    """schedule 응답 구조가 바뀌어도 gameId/gmkey가 있는 dict를 재귀적으로 찾는다."""
    games: list[dict] = []

    if isinstance(obj, dict):
        if any(k in obj for k in ("gameId", "gameID", "game_id", "gmkey")):
            games.append(obj)

        for value in obj.values():
            games.extend(find_game_objects(value))

    elif isinstance(obj, list):
        for item in obj:
            games.extend(find_game_objects(item))

    return games


def get_game_id(g: dict) -> str:
    return str(
        pick(
            g,
            "gameId",
            "gameID",
            "game_id",
            "gmkey",
            default="",
        )
        or ""
    )


def get_away_team_code(g: dict) -> str | None:
    return pick(
        g,
        "aCode",
        "awayTeamCode",
        "awayCode",
        "away_team_code",
    )


def get_home_team_code(g: dict) -> str | None:
    return pick(
        g,
        "hCode",
        "homeTeamCode",
        "homeCode",
        "home_team_code",
    )


def get_away_team_name(g: dict) -> str | None:
    return pick(
        g,
        "aName",
        "awayTeamName",
        "awayName",
        "away_team_name",
    )


def get_home_team_name(g: dict) -> str | None:
    return pick(
        g,
        "hName",
        "homeTeamName",
        "homeName",
        "home_team_name",
    )


def get_away_team_full_name(g: dict) -> str | None:
    return pick(
        g,
        "aFullName",
        "awayTeamFullName",
        "awayFullName",
        "away_team_full_name",
    )


def get_home_team_full_name(g: dict) -> str | None:
    return pick(
        g,
        "hFullName",
        "homeTeamFullName",
        "homeFullName",
        "home_team_full_name",
    )


def get_game_date(g: dict, requested_date: str) -> str:
    value = pick(
        g,
        "gdate",
        "gameDate",
        "game_date",
        "date",
        default=requested_date,
    )

    if value is None:
        return requested_date

    value = str(value)

    # 2025-04-30 같은 형식이면 20250430으로 변환
    if "-" in value:
        return value.replace("-", "")[:8]

    return value[:8]


def get_game_time(g: dict) -> str | None:
    value = pick(
        g,
        "gtime",
        "gameTime",
        "game_time",
        "time",
        "gameDateTime",
    )

    if value is None:
        return None

    return str(value)


def get_score(g: dict, side: str):
    """side: away 또는 home."""
    score = g.get("score", {}) or {}

    if side == "away":
        return pick(
            score,
            "aScore",
            "awayScore",
            "awayTeamScore",
            default=pick(
                g,
                "awayScore",
                "awayTeamScore",
                "aScore",
            ),
        )

    if side == "home":
        return pick(
            score,
            "hScore",
            "homeScore",
            "homeTeamScore",
            default=pick(
                g,
                "homeScore",
                "homeTeamScore",
                "hScore",
            ),
        )

    return None


def is_real_kbo_team_code(code: str | None, year: int | None = None) -> bool:
    """실제 KBO 팀 코드인지 확인.

    config.REAL_KBO_TEAM_CODES를 우선 사용하고,
    연도별 팀 코드 함수가 있으면 그것도 함께 사용한다.
    """
    if not code:
        return False

    code = str(code)

    valid_codes = set(getattr(config, "REAL_KBO_TEAM_CODES", []))

    if year is not None and hasattr(config, "team_codes_for_year"):
        try:
            valid_codes.update(config.team_codes_for_year(year))
        except Exception:  # noqa: BLE001
            pass

    return code in valid_codes


def is_valid_kbo_game_object(g: dict, requested_date: str) -> bool:
    """정상 KBO 경기 객체인지 검증."""
    game_id = get_game_id(g)

    if not game_id:
        return False

    # 요청 날짜와 game_id 앞 8자리가 반드시 같아야 함
    # 현재 경기/다른 날짜 경기 혼입 방지
    if not game_id.startswith(requested_date):
        return False

    year = int(requested_date[:4])

    away_code = get_away_team_code(g)
    home_code = get_home_team_code(g)

    if not is_real_kbo_team_code(away_code, year):
        return False

    if not is_real_kbo_team_code(home_code, year):
        return False

    return True


def game_object_to_row(g: dict, requested_date: str) -> dict:
    """네이버 경기 객체를 games.csv row로 변환."""
    return {
        "game_id": get_game_id(g),
        "game_date": get_game_date(g, requested_date),
        "game_time": get_game_time(g),
        "weekday": pick(g, "gweek", "weekday", "dayOfWeek"),
        "away_team_code": get_away_team_code(g),
        "away_team_name": get_away_team_name(g),
        "away_team_full_name": get_away_team_full_name(g),
        "home_team_code": get_home_team_code(g),
        "home_team_name": get_home_team_name(g),
        "home_team_full_name": get_home_team_full_name(g),
        "stadium": pick(g, "stadium", "stadiumName", "ground", "groundName"),
        "status_code": pick(g, "statusCode", "gameStatusCode", "status_code"),
        "status_info": pick(g, "statusInfo", "gameStatusInfo", "status_info"),
        "cancel_flag": pick(g, "cancelFlag", "canceled", "cancel", "cancel_flag"),
        "suspended_flag": pick(g, "suspended", "suspendedFlag", "suspended_flag"),
        "dheader": pick(g, "dheader", "doubleHeader", "doubleHeaderFlag"),
        "away_score": get_score(g, "away"),
        "home_score": get_score(g, "home"),
    }


# ------------------------------------------------------------
# record fallback
# ------------------------------------------------------------
def generate_game_id_candidates(date_yyyymmdd: str, team_codes: list[str]) -> list[str]:
    """record fallback용 game_id 후보 생성.

    game_id 예상 형식:
    YYYYMMDD + awayCode + homeCode + suffix

    예:
    20250430LTSK0

    더블헤더/특수 케이스 대비로 suffix 0, 1, 2를 모두 시도.
    """
    candidates: list[str] = []
    suffixes = ["0", "1", "2"]

    for away in team_codes:
        for home in team_codes:
            if away == home:
                continue

            for suffix in suffixes:
                candidates.append(f"{date_yyyymmdd}{away}{home}{suffix}")

    return candidates


def extract_games_from_record(record_json: dict, requested_date: str | None = None) -> list[dict]:
    """record API 응답에서 경기 목록 추출."""
    rd = record_json.get("result", {}).get("recordData", {}) or {}

    games = (
        rd.get("games")
        or rd.get("gameList")
        or rd.get("schedule")
        or []
    )

    rows: list[dict] = []

    for g in games:
        game_id = get_game_id(g)

        if not game_id:
            continue

        date = requested_date or game_id[:8]

        rows.append(game_object_to_row(g, date))

    return rows


def collect_games_from_record_fallback(
    client: NaverClient,
    date_yyyymmdd: str,
) -> list[dict]:
    """schedule에서 못 찾았을 때 record API로 그날 경기 목록을 복구."""
    year = int(date_yyyymmdd[:4])

    if hasattr(config, "team_codes_for_year"):
        team_codes = config.team_codes_for_year(year)
    else:
        team_codes = list(getattr(config, "REAL_KBO_TEAM_CODES", []))

    candidates = generate_game_id_candidates(date_yyyymmdd, team_codes)

    for game_id in candidates:
        try:
            record_json, from_cache = client.fetch_record_cached(game_id)
            games = extract_games_from_record(record_json, requested_date=date_yyyymmdd)

            if games:
                valid_games = [
                    g for g in games
                    if str(g.get("game_id", "")).startswith(date_yyyymmdd)
                    and is_real_kbo_team_code(g.get("away_team_code"), year)
                    and is_real_kbo_team_code(g.get("home_team_code"), year)
                ]

                if valid_games:
                    logger.info(
                        "  [fallback success] seed=%s, games=%d",
                        game_id,
                        len(valid_games),
                    )
                    return valid_games

            if not from_cache:
                client.polite_sleep()

        except Exception:
            # 존재하지 않는 game_id 후보는 조용히 스킵
            pass

    logger.warning("  [fallback failed] %s", date_yyyymmdd)

    return []


# ------------------------------------------------------------
# games.csv 입출력
# ------------------------------------------------------------
def empty_games_df() -> pd.DataFrame:
    return pd.DataFrame(columns=GAME_COLUMNS)


def read_existing_games_csv(path: Path) -> pd.DataFrame:
    """기존 games.csv를 안전하게 읽는다."""
    if not path.exists():
        return empty_games_df()

    try:
        df = pd.read_csv(path, dtype={"game_id": str})
    except EmptyDataError:
        return empty_games_df()

    if df.empty:
        return empty_games_df()

    # 누락 컬럼 보강
    for col in GAME_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[GAME_COLUMNS]


def normalize_games_df(df: pd.DataFrame) -> pd.DataFrame:
    """games.csv 저장 전 정리."""
    if df.empty:
        return empty_games_df()

    for col in GAME_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[GAME_COLUMNS].copy()

    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = df["game_date"].astype(str)

    df = (
        df
        .drop_duplicates(subset=["game_id"], keep="last")
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )

    return df


# ------------------------------------------------------------
# 메인 수집
# ------------------------------------------------------------
def collect_games_for_date(client: NaverClient, date: str) -> list[dict]:
    """특정 날짜의 경기 목록 수집."""
    rows: list[dict] = []

    # 1차: schedule API
    try:
        data, from_cache = client.fetch_schedule_cached(date)

        raw_game_objects = find_game_objects(data)

        valid_game_objects = [
            g for g in raw_game_objects
            if is_valid_kbo_game_object(g, date)
        ]

        rows = [
            game_object_to_row(g, date)
            for g in valid_game_objects
        ]

        if not from_cache:
            client.polite_sleep()

        logger.info(
            "[games] %s schedule ok, raw=%d, valid=%d (cache=%s)",
            date,
            len(raw_game_objects),
            len(rows),
            from_cache,
        )

        if raw_game_objects and not rows:
            sample = raw_game_objects[0]
            logger.warning(
                "[games] %s raw objects found but valid=0. sample keys=%s",
                date,
                list(sample.keys()),
            )
            logger.warning(
                "[games] %s sample gameId=%s away=%s home=%s",
                date,
                get_game_id(sample),
                get_away_team_code(sample),
                get_home_team_code(sample),
            )

    except Exception as e:  # noqa: BLE001
        logger.warning("[games] %s schedule failed: %s", date, e)
        log_failed_request("schedule", date, repr(e))

    # 2차: record fallback
    if not rows:
        rows = collect_games_from_record_fallback(client, date)

        if not rows:
            # 월요일/우천취소/비시즌 등 경기 없는 날일 수 있음
            logger.info("[games] %s no games found", date)

    return rows


def collect_games(
    start_date: str,
    end_date: str,
    client: NaverClient | None = None,
    team_codes: list[str] | None = None,
) -> pd.DataFrame:
    """기간 내 모든 KBO 경기 목록 수집 -> games.csv.

    기존 games.csv와 merge해서 저장한다.
    """
    ensure_dirs()

    client = client or NaverClient()

    all_rows: list[dict] = []

    for date in daterange(start_date, end_date):
        try:
            date_rows = collect_games_for_date(client, date)
        except Exception as e:  # noqa: BLE001
            log_failed_request("schedule", date, repr(e))
            logger.exception("[games] %s failed: %s", date, e)
            continue

        if team_codes:
            date_rows = [
                r for r in date_rows
                if r.get("away_team_code") in team_codes
                or r.get("home_team_code") in team_codes
            ]

        all_rows.extend(date_rows)

    new_df = pd.DataFrame(all_rows)

    if new_df.empty:
        new_df = empty_games_df()
    else:
        new_df = normalize_games_df(new_df)

    out_path = config.PROCESSED_DIR / "games.csv"

    old_df = read_existing_games_csv(out_path)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = normalize_games_df(merged)

    # merged가 비어 있어도 헤더는 남긴다.
    # 그래야 다음 pd.read_csv에서 EmptyDataError가 안 난다.
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info(
        "[saved] %s rows=%d (+%d new range rows)",
        out_path,
        len(merged),
        len(new_df),
    )

    return merged