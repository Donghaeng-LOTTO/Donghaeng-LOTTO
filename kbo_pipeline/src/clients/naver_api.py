"""네이버 스포츠 API 클라이언트.

- requests.Session 재사용
- 재시도 + 백오프
- raw JSON 캐시 (이미 저장된 파일은 재요청하지 않음)
- 실패 로그를 data/logs/failed_requests.csv 에 기록
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from src import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
def ensure_dirs() -> None:
    for path in config.ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log_failed_request(kind: str, key: str, error: str) -> None:
    """kind: schedule / record / relay, key: date 또는 game_id"""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = config.FAILED_REQUESTS_CSV.exists()

    with config.FAILED_REQUESTS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "kind", "key", "error"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            kind,
            key,
            str(error)[:500],
        ])


def to_hyphen_date(date_yyyymmdd: str) -> str:
    return f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"


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

    # --- 저수준 GET (재시도 포함) ---
    def _get_json(self, url: str, params: dict | None = None, referer: str | None = None) -> dict:
        headers = {}
        if referer:
            headers["referer"] = referer

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                res = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
                # 400(잘못된 파라미터) · 404(없는 리소스)는 재시도해도 의미 없으므로 즉시 실패
                if res.status_code == 404:
                    raise FileNotFoundError(f"404 Not Found: {res.url}")
                if res.status_code == 400:
                    raise requests.HTTPError(
                        f"400 Bad Request: {res.url}", response=res
                    )
                res.raise_for_status()
                return res.json()
            except (FileNotFoundError, requests.HTTPError) as e:
                if isinstance(e, requests.HTTPError) and (
                    e.response is not None and e.response.status_code == 400
                ):
                    raise  # 즉시 재전파 (재시도 없음)
                if isinstance(e, FileNotFoundError):
                    raise
                last_error = e
                wait = config.RETRY_BACKOFF_SEC * attempt
                logger.warning("GET failed (%s/%s) %s params=%s err=%s -> retry in %.1fs",
                               attempt, self.max_retries, url, params, e, wait)
                time.sleep(wait)
                continue
            except Exception as e:  # noqa: BLE001
                last_error = e
                wait = config.RETRY_BACKOFF_SEC * attempt
                logger.warning("GET failed (%s/%s) %s params=%s err=%s -> retry in %.1fs",
                               attempt, self.max_retries, url, params, e, wait)
                time.sleep(wait)

        raise RuntimeError(f"GET failed after {self.max_retries} retries: {url} / {last_error}")

    def polite_sleep(self) -> None:
        time.sleep(self.sleep_sec)

    # --- schedule ---
    def fetch_schedule(self, date_yyyymmdd: str) -> dict:
        """파라미터 조합이 시즌/시점에 따라 다를 수 있어 여러 조합을 순서대로 시도."""
        date_hyphen = to_hyphen_date(date_yyyymmdd)
        param_candidates = [
            {"upperCategoryId": "kbaseball", "date": date_yyyymmdd},
            {"upperCategoryId": "kbaseball", "date": date_hyphen},
            {"categoryId": "kbo", "date": date_yyyymmdd},
            {"categoryId": "kbo", "date": date_hyphen},
            {"upperCategoryId": "kbaseball", "categoryId": "kbo", "date": date_yyyymmdd},
            {"upperCategoryId": "kbaseball", "categoryId": "kbo", "date": date_hyphen},
        ]

        last_error = None
        for params in param_candidates:
            try:
                return self._get_json(config.SCHEDULE_URL, params=params)
            except Exception as e:  # noqa: BLE001
                last_error = e

        raise RuntimeError(f"schedule fetch failed: {date_yyyymmdd} / {last_error}")

    def fetch_schedule_cached(self, date_yyyymmdd: str) -> tuple[dict, bool]:
        """returns (data, from_cache)"""
        cache_path = config.RAW_SCHEDULE_DIR / f"{date_yyyymmdd}_schedule.json"
        if cache_path.exists():
            return load_json(cache_path), True

        data = self.fetch_schedule(date_yyyymmdd)
        save_json(data, cache_path)
        return data, False

    # --- record ---
    def fetch_record(self, game_id: str) -> dict:
        url = config.RECORD_URL.format(game_id=game_id)
        data = self._get_json(
            url, referer=f"https://m.sports.naver.com/game/{game_id}/record"
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

    # --- relay ---
    def fetch_relay(self, game_id: str, inning: int | None = None) -> dict:
        url = config.RELAY_URL.format(game_id=game_id)
        params = {"inning": inning} if inning is not None else None
        return self._get_json(
            url, params=params,
            referer=f"https://m.sports.naver.com/game/{game_id}/relay",
        )

    def relay_cache_path(self, game_id: str, inning: int | None = None) -> Path:
        d = config.RAW_RELAY_DIR / game_id
        if inning is None:
            return d / f"{game_id}_base.json"
        return d / f"{game_id}_inning_{inning:02d}.json"

    def fetch_relay_cached(self, game_id: str, inning: int | None = None) -> tuple[dict, bool]:
        cache_path = self.relay_cache_path(game_id, inning)
        if cache_path.exists():
            return load_json(cache_path), True

        data = self.fetch_relay(game_id, inning)
        save_json(data, cache_path)
        return data, False
