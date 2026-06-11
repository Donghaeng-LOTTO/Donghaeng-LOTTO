"""games.csv 수집.

1차: schedule API
2차: schedule 실패/유효 경기 0개 -> record fallback (game_id 후보를 record에 던져
     recordData.games에서 그날 전체 경기 목록 획득)

핵심 검증:
- game_id 앞 8자리 == 요청 날짜
- 비정규 항목(BMBC, CHITCHIT, SPORTSN 등) 제외
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from src import config
from src.clients.naver_api import (
    NaverClient, ensure_dirs, save_json, log_failed_request,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def daterange(start_date: str, end_date: str):
    cur = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def find_game_objects(obj) -> list[dict]:
    """schedule 응답 구조가 바뀌어도 gameId가 있는 dict를 재귀적으로 찾는다."""
    games = []
    if isinstance(obj, dict):
        if "gameId" in obj:
            games.append(obj)
        for value in obj.values():
            games.extend(find_game_objects(value))
    elif isinstance(obj, list):
        for item in obj:
            games.extend(find_game_objects(item))
    return games


def is_valid_kbo_game_object(g: dict, requested_date: str) -> bool:
    game_id = str(g.get("gameId") or g.get("gmkey") or "")
    if not game_id:
        return False
    # 요청 날짜와 game_id 앞 8자리가 반드시 같아야 함 (현재 경기 혼입 방지)
    if not game_id.startswith(requested_date):
        return False
    if g.get("aCode") not in config.REAL_KBO_TEAM_CODES:
        return False
    if g.get("hCode") not in config.REAL_KBO_TEAM_CODES:
        return False
    return True


def game_object_to_row(g: dict, requested_date: str) -> dict:
    score = g.get("score", {}) or {}
    return {
        "game_id": g.get("gameId") or g.get("gmkey"),
        "game_date": g.get("gdate") or requested_date,
        "game_time": g.get("gtime"),
        "weekday": g.get("gweek"),
        "away_team_code": g.get("aCode"),
        "away_team_name": g.get("aName"),
        "away_team_full_name": g.get("aFullName"),
        "home_team_code": g.get("hCode"),
        "home_team_name": g.get("hName"),
        "home_team_full_name": g.get("hFullName"),
        "stadium": g.get("stadium"),
        "status_code": g.get("statusCode"),
        "cancel_flag": g.get("cancelFlag"),
        "dheader": g.get("dheader"),
        "away_score": score.get("aScore"),
        "home_score": score.get("hScore"),
    }


# ------------------------------------------------------------
# record fallback
# ------------------------------------------------------------
def generate_game_id_candidates(date_yyyymmdd: str, team_codes: list[str]) -> list[str]:
    """game_id 형식: YYYYMMDD + awayCode + homeCode + 0 (더블헤더는 끝자리 1,2) 더블헤더/특수 케이스 대비: 끝자리 1, 2도 후보에 포함."""
    candidates = []

    suffixes = ["0", "1", "2"]

    for away in team_codes:
        for home in team_codes:
            if away == home:
                continue
            for suffix in suffixes:
                candidates.append(f"{date_yyyymmdd}{away}{home}{suffix}")
    return candidates


def extract_games_from_record(record_json: dict) -> list[dict]:
    rd = record_json.get("result", {}).get("recordData", {}) or {}
    games = rd.get("games", []) or []

    rows = []
    for g in games:
        game_id = g.get("gameId") or g.get("gmkey")
        if not game_id:
            continue
        score = g.get("score", {}) or {}
        rows.append({
            "game_id": game_id,
            "game_date": g.get("gdate"),
            "game_time": g.get("gtime"),
            "weekday": g.get("gweek"),
            "away_team_code": g.get("aCode"),
            "away_team_name": g.get("aName"),
            "away_team_full_name": g.get("aFullName"),
            "home_team_code": g.get("hCode"),
            "home_team_name": g.get("hName"),
            "home_team_full_name": g.get("hFullName"),
            "stadium": g.get("stadium"),
            "status_code": g.get("statusCode"),
            "cancel_flag": g.get("cancelFlag"),
            "dheader": g.get("dheader"),
            "away_score": score.get("aScore"),
            "home_score": score.get("hScore"),
        })
    return rows


def collect_games_from_record_fallback(
    client: NaverClient, date_yyyymmdd: str
) -> list[dict]:
    year = int(date_yyyymmdd[:4])
    team_codes = config.team_codes_for_year(year)
    candidates = generate_game_id_candidates(date_yyyymmdd, team_codes)

    for game_id in candidates:
        try:
            record_json, from_cache = client.fetch_record_cached(game_id)
            games = extract_games_from_record(record_json)
            if games:
                logger.info("  [fallback success] seed=%s, games=%d", game_id, len(games))
                # record.games에도 다른 날짜 항목이 섞일 수 있으므로 재검증
                return [
                    g for g in games
                    if str(g["game_id"]).startswith(date_yyyymmdd)
                    and g.get("away_team_code") in config.REAL_KBO_TEAM_CODES
                    and g.get("home_team_code") in config.REAL_KBO_TEAM_CODES
                ]
            if not from_cache:
                client.polite_sleep()
        except Exception:  # noqa: BLE001
            # 존재하지 않는 game_id 후보는 조용히 스킵
            pass

    logger.warning("  [fallback failed] %s", date_yyyymmdd)
    return []


# ------------------------------------------------------------
# 메인 수집
# ------------------------------------------------------------
def collect_games_for_date(client: NaverClient, date: str) -> list[dict]:
    rows: list[dict] = []

    # 1차: schedule API (캐시 우선)
    try:
        data, from_cache = client.fetch_schedule_cached(date)
        game_objects = [
            g for g in find_game_objects(data)
            if is_valid_kbo_game_object(g, date)
        ]
        rows = [game_object_to_row(g, date) for g in game_objects]
        if not from_cache:
            client.polite_sleep()
        logger.info("[games] %s schedule ok, valid=%d (cache=%s)", date, len(rows), from_cache)
    except Exception as e:  # noqa: BLE001
        logger.warning("[games] %s schedule failed: %s", date, e)
        log_failed_request("schedule", date, repr(e))

    # 2차: record fallback
    if not rows:
        rows = collect_games_from_record_fallback(client, date)
        if not rows:
            # 경기 없는 날(월요일 등)일 수 있으므로 에러는 아니지만 흔적은 남긴다
            logger.info("[games] %s no games found", date)

    return rows


def collect_games(
    start_date: str,
    end_date: str,
    client: NaverClient | None = None,
    team_codes: list[str] | None = None,
) -> pd.DataFrame:
    """기간 내 모든 KBO 경기 목록 수집 -> games.csv (기존 파일과 merge)"""
    ensure_dirs()
    client = client or NaverClient()

    all_rows: list[dict] = []
    for date in daterange(start_date, end_date):
        try:
            date_rows = collect_games_for_date(client, date)
        except Exception as e:  # noqa: BLE001
            log_failed_request("schedule", date, repr(e))
            continue

        if team_codes:
            date_rows = [
                r for r in date_rows
                if r.get("away_team_code") in team_codes
                or r.get("home_team_code") in team_codes
            ]
        all_rows.extend(date_rows)

    new_df = pd.DataFrame(all_rows)

    out_path = config.PROCESSED_DIR / "games.csv"
    if out_path.exists():
        old_df = pd.read_csv(out_path, dtype={"game_id": str})
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    if not merged.empty:
        merged["game_id"] = merged["game_id"].astype(str)
        merged = (
            merged
            .drop_duplicates(subset=["game_id"], keep="last")
            .sort_values(["game_date", "game_id"])
            .reset_index(drop=True)
        )

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s rows=%d (+%d new range rows)", out_path, len(merged), len(new_df))
    return merged
