"""네이버 스포츠 API 클라이언트.

- requests.Session 재사용
- 재시도 + 백오프
- raw JSON 캐시
- schedule 응답 날짜 검증
- 잘못된 날짜 응답은 캐시하지 않음
- 실패 로그를 data/logs/failed_requests.csv 에 기록
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from src import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 네이버 스포츠 API URL
# ------------------------------------------------------------
NAVER_SCHEDULE_GATEWAY_URL = "https://api-gw.sports.naver.com/schedule/games"
NAVER_KBO_SCHEDULE_REFERER = "https://m.sports.naver.com/kbaseball/schedule/index"


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
def ensure_dirs() -> None:
    for path in config.ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log_failed_request(kind: str, key: str, error: str) -> None:
    """kind: schedule / record / relay, key: date 또는 game_id"""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = config.FAILED_REQUESTS_CSV.exists()

    with config.FAILED_REQUESTS_CSV.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "kind", "key", "error"])

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                kind,
                key,
                str(error)[:500],
            ]
        )


def to_hyphen_date(date_yyyymmdd: str) -> str:
    """20250404 -> 2025-04-04"""
    return f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"


def normalize_yyyymmdd(value: Any) -> str | None:
    """날짜처럼 보이는 값을 YYYYMMDD로 정규화.

    예:
    - 2025-04-04 -> 20250404
    - 20250404 -> 20250404
    - 2025-04-04T18:30:00 -> 20250404
    """
    if value is None:
        return None

    s = str(value)
    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) >= 8:
        return digits[:8]

    return None


def find_game_ids(obj: Any) -> list[str]:
    """응답 JSON 전체에서 gameId/gmkey 값을 재귀적으로 찾는다."""
    game_ids: list[str] = []

    if isinstance(obj, dict):
        for key in ("gameId", "gameID", "game_id", "gmkey"):
            value = obj.get(key)
            if value:
                game_ids.append(str(value))

        for value in obj.values():
            game_ids.extend(find_game_ids(value))

    elif isinstance(obj, list):
        for item in obj:
            game_ids.extend(find_game_ids(item))

    return game_ids


def find_date_values(obj: Any) -> list[str]:
    """응답 JSON 전체에서 날짜처럼 보이는 값을 재귀적으로 찾는다."""
    date_keys = {
        "date",
        "gameDate",
        "game_date",
        "gdate",
        "gameDateTime",
        "game_datetime",
    }

    values: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in date_keys:
                normalized = normalize_yyyymmdd(value)
                if normalized:
                    values.append(normalized)

            values.extend(find_date_values(value))

    elif isinstance(obj, list):
        for item in obj:
            values.extend(find_date_values(item))

    return values


def schedule_response_matches_date(data: Any, requested_date: str) -> bool:
    """schedule 응답이 요청 날짜와 맞는지 검증.

    날짜 파라미터가 안 먹으면 네이버가 오늘 경기 목록을 줄 수 있다.
    그래서 gameId가 있는데 요청 날짜로 시작하지 않으면 잘못된 응답으로 판단한다.
    """
    game_ids = find_game_ids(data)

    if game_ids:
        return any(game_id.startswith(requested_date) for game_id in game_ids)

    date_values = find_date_values(data)

    if requested_date in date_values:
        return True

    # gameId도 없고 날짜도 없으면 경기 없는 날일 수 있으므로 일단 허용
    return True


# ------------------------------------------------------------
# 클라이언트
# ------------------------------------------------------------
class NaverClient:
    def __init__(
        self,
        sleep_sec: float = config.DEFAULT_SLEEP_SEC,
        max_retries: int = config.MAX_RETRIES,
        timeout: int = config.REQUEST_TIMEOUT,
    ):
        self.session = requests.Session()
        self.session.headers.update(config.HEADERS)

        self.sleep_sec = sleep_sec
        self.max_retries = max_retries
        self.timeout = timeout

    # --------------------------------------------------------
    # 저수준 GET
    # --------------------------------------------------------
    def _get_json(
        self,
        url: str,
        params: dict | None = None,
        referer: str | None = None,
    ) -> dict:
        headers = {}

        if referer:
            headers["Referer"] = referer

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                res = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )

                logger.debug(
                    "GET %s params=%s -> status=%s final_url=%s",
                    url,
                    params,
                    res.status_code,
                    res.url,
                )

                if res.status_code == 404:
                    raise FileNotFoundError(f"404 Not Found: {res.url}")

                if res.status_code == 400:
                    raise requests.HTTPError(
                        f"400 Bad Request: {res.url}",
                        response=res,
                    )

                res.raise_for_status()
                return res.json()

            except FileNotFoundError:
                raise

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    raise

                last_error = e
                wait = config.RETRY_BACKOFF_SEC * attempt

                logger.warning(
                    "GET failed (%s/%s) %s params=%s err=%s -> retry in %.1fs",
                    attempt,
                    self.max_retries,
                    url,
                    params,
                    e,
                    wait,
                )

                time.sleep(wait)

            except Exception as e:  # noqa: BLE001
                last_error = e
                wait = config.RETRY_BACKOFF_SEC * attempt

                logger.warning(
                    "GET failed (%s/%s) %s params=%s err=%s -> retry in %.1fs",
                    attempt,
                    self.max_retries,
                    url,
                    params,
                    e,
                    wait,
                )

                time.sleep(wait)

        raise RuntimeError(
            f"GET failed after {self.max_retries} retries: {url} / {last_error}"
        )

    def polite_sleep(self) -> None:
        time.sleep(self.sleep_sec)

    # --------------------------------------------------------
    # schedule
    # --------------------------------------------------------
    def _schedule_request_candidates(
        self,
        date_yyyymmdd: str,
    ) -> list[tuple[str, dict]]:
        """schedule API 후보 URL/params 생성.

        date= 하나만 쓰면 현재 날짜 경기가 나오는 경우가 있어서
        fromDate/toDate 조합을 먼저 시도한다.
        """
        date_hyphen = to_hyphen_date(date_yyyymmdd)

        urls: list[str] = [NAVER_SCHEDULE_GATEWAY_URL]

        config_schedule_url = getattr(config, "SCHEDULE_URL", None)
        if config_schedule_url and config_schedule_url not in urls:
            urls.append(config_schedule_url)

        param_candidates = [
            # 1순위: 기간 검색 방식, 하이픈 날짜
            {
                "upperCategoryId": "kbaseball",
                "fromDate": date_hyphen,
                "toDate": date_hyphen,
            },
            {
                "upperCategoryId": "kbaseball",
                "categoryId": "kbo",
                "fromDate": date_hyphen,
                "toDate": date_hyphen,
            },
            {
                "categoryId": "kbo",
                "fromDate": date_hyphen,
                "toDate": date_hyphen,
            },

            # 2순위: 기간 검색 방식, YYYYMMDD 날짜
            {
                "upperCategoryId": "kbaseball",
                "fromDate": date_yyyymmdd,
                "toDate": date_yyyymmdd,
            },
            {
                "upperCategoryId": "kbaseball",
                "categoryId": "kbo",
                "fromDate": date_yyyymmdd,
                "toDate": date_yyyymmdd,
            },
            {
                "categoryId": "kbo",
                "fromDate": date_yyyymmdd,
                "toDate": date_yyyymmdd,
            },

            # 3순위: 단일 date 방식, 하이픈 날짜
            {
                "upperCategoryId": "kbaseball",
                "date": date_hyphen,
            },
            {
                "upperCategoryId": "kbaseball",
                "categoryId": "kbo",
                "date": date_hyphen,
            },
            {
                "categoryId": "kbo",
                "date": date_hyphen,
            },

            # 4순위: 단일 date 방식, YYYYMMDD 날짜
            {
                "upperCategoryId": "kbaseball",
                "date": date_yyyymmdd,
            },
            {
                "upperCategoryId": "kbaseball",
                "categoryId": "kbo",
                "date": date_yyyymmdd,
            },
            {
                "categoryId": "kbo",
                "date": date_yyyymmdd,
            },

            # 5순위: 혹시 다른 이름으로 받는 경우 대비
            {
                "upperCategoryId": "kbaseball",
                "startDate": date_hyphen,
                "endDate": date_hyphen,
            },
            {
                "upperCategoryId": "kbaseball",
                "categoryId": "kbo",
                "startDate": date_hyphen,
                "endDate": date_hyphen,
            },
            {
                "categoryId": "kbo",
                "startDate": date_hyphen,
                "endDate": date_hyphen,
            },
            {
                "upperCategoryId": "kbaseball",
                "startDate": date_yyyymmdd,
                "endDate": date_yyyymmdd,
            },
            {
                "upperCategoryId": "kbaseball",
                "categoryId": "kbo",
                "startDate": date_yyyymmdd,
                "endDate": date_yyyymmdd,
            },
            {
                "categoryId": "kbo",
                "startDate": date_yyyymmdd,
                "endDate": date_yyyymmdd,
            },
        ]

        candidates: list[tuple[str, dict]] = []

        for url in urls:
            for params in param_candidates:
                candidates.append((url, params))

        return candidates

    def fetch_schedule(self, date_yyyymmdd: str) -> dict:
        """특정 날짜 schedule JSON 요청.

        잘못된 날짜 응답이 오면 버리고 다음 후보를 시도한다.
        """
        last_error: Exception | None = None
        rejected_samples: list[str] = []

        for url, params in self._schedule_request_candidates(date_yyyymmdd):
            try:
                data = self._get_json(
                    url,
                    params=params,
                    referer=NAVER_KBO_SCHEDULE_REFERER,
                )

                if schedule_response_matches_date(data, date_yyyymmdd):
                    logger.info(
                        "[schedule fetch ok] date=%s url=%s params=%s",
                        date_yyyymmdd,
                        url,
                        params,
                    )
                    return data

                game_ids = find_game_ids(data)
                sample = game_ids[:5]

                rejected_samples.append(
                    f"url={url}, params={params}, sample_game_ids={sample}"
                )

                logger.warning(
                    "[schedule rejected: wrong date] requested=%s url=%s params=%s sample_game_ids=%s",
                    date_yyyymmdd,
                    url,
                    params,
                    sample,
                )

            except Exception as e:  # noqa: BLE001
                last_error = e

                logger.debug(
                    "[schedule candidate failed] date=%s url=%s params=%s err=%s",
                    date_yyyymmdd,
                    url,
                    params,
                    e,
                )

                continue

        raise RuntimeError(
            "schedule fetch failed: "
            f"{date_yyyymmdd} / last_error={last_error} "
            f"/ rejected={rejected_samples[:3]}"
        )

    def fetch_schedule_cached(self, date_yyyymmdd: str) -> tuple[dict, bool]:
        """returns (data, from_cache).

        캐시가 있어도 요청 날짜와 안 맞으면 삭제하고 다시 요청한다.
        """
        cache_path = config.RAW_SCHEDULE_DIR / f"{date_yyyymmdd}_schedule.json"

        if cache_path.exists():
            try:
                cached_data = load_json(cache_path)

                if schedule_response_matches_date(cached_data, date_yyyymmdd):
                    return cached_data, True

                logger.warning(
                    "[schedule cache ignored] wrong date cache=%s requested=%s sample_game_ids=%s",
                    cache_path,
                    date_yyyymmdd,
                    find_game_ids(cached_data)[:5],
                )

                cache_path.unlink(missing_ok=True)

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[schedule cache read failed] cache=%s err=%s -> refetch",
                    cache_path,
                    e,
                )
                cache_path.unlink(missing_ok=True)

        data = self.fetch_schedule(date_yyyymmdd)
        save_json(data, cache_path)

        return data, False

    # --------------------------------------------------------
    # record
    # --------------------------------------------------------
    def fetch_record(self, game_id: str) -> dict:
        url = config.RECORD_URL.format(game_id=game_id)

        data = self._get_json(
            url,
            referer=f"https://m.sports.naver.com/game/{game_id}/record",
        )

        if not data.get("success"):
            raise ValueError(f"record success=false: {game_id}")

        return data

    def fetch_record_cached(self, game_id: str) -> tuple[dict, bool]:
        cache_path = config.RAW_RECORD_DIR / game_id / f"{game_id}_record.json"

        if cache_path.exists():
            return load_json(cache_path), True

        data = self.fetch_record(game_id)
        save_json(data, cache_path)

        return data, False

    # --------------------------------------------------------
    # relay
    # --------------------------------------------------------
    def fetch_relay(self, game_id: str, inning: int | None = None) -> dict:
        url = config.RELAY_URL.format(game_id=game_id)
        params = {"inning": inning} if inning is not None else None

        return self._get_json(
            url,
            params=params,
            referer=f"https://m.sports.naver.com/game/{game_id}/relay",
        )

    def relay_cache_path(self, game_id: str, inning: int | None = None) -> Path:
        d = config.RAW_RELAY_DIR / game_id

        if inning is None:
            return d / f"{game_id}_base.json"

        return d / f"{game_id}_inning_{inning:02d}.json"

    def fetch_relay_cached(
        self,
        game_id: str,
        inning: int | None = None,
    ) -> tuple[dict, bool]:
        cache_path = self.relay_cache_path(game_id, inning)

        if cache_path.exists():
            return load_json(cache_path), True

        data = self.fetch_relay(game_id, inning)
        save_json(data, cache_path)

        return data, False