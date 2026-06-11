"""2008~2025 KBO 전체 경기 수집 (연/월 단위, 재시작 가능).

순서:
1) schedule(+record fallback) -> games.csv
2) record -> games_detail / boxscores
3) relay -> raw JSON + naver_players_seen.csv

이미 저장된 raw JSON은 재요청하지 않으므로 중단 후 다시 실행해도 안전하다.

사용 예:
  uv run python scripts/run_collect_2008_2025.py
  uv run python scripts/run_collect_2008_2025.py --start-year 2010 --end-year 2010
  uv run python scripts/run_collect_2008_2025.py --start-year 2008 --end-year 2008 --months 6
  uv run python scripts/run_collect_2008_2025.py --skip-relay
"""
from __future__ import annotations

import argparse
import calendar
import logging
import sys
from pathlib import Path

# 프로젝트 루트에서 실행할 수 있도록 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.clients.naver_api import NaverClient, ensure_dirs  # noqa: E402
from src.collectors.collect_games import collect_games  # noqa: E402
from src.collectors.collect_records import collect_records  # noqa: E402
from src.collectors.collect_relays import collect_relays  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_collect")


def month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}01", f"{year}{month:02d}{last_day:02d}"


def season_months(year: int, months: list[int] | None) -> list[int]:
    if months:
        return months
    start_m = int(config.SEASON_START_MMDD[:2])
    end_m = int(config.SEASON_END_MMDD[:2])
    return list(range(start_m, end_m + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", type=int, nargs="*", default=None,
                        help="특정 월만 수집 (예: --months 6 7)")
    parser.add_argument("--sleep-sec", type=float, default=config.DEFAULT_SLEEP_SEC)
    parser.add_argument("--team-codes", type=str, nargs="*", default=None,
                        help="특정 팀 경기만 (예: --team-codes LT)")
    parser.add_argument("--skip-games", action="store_true")
    parser.add_argument("--skip-records", action="store_true")
    parser.add_argument("--skip-relay", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    client = NaverClient(sleep_sec=args.sleep_sec)

    # ----------------------------------------------------------
    # 1) games.csv : 연/월 단위 수집
    # ----------------------------------------------------------
    if not args.skip_games:
        for year in range(args.start_year, args.end_year + 1):
            for month in season_months(year, args.months):
                start, end = month_range(year, month)
                logger.info("=" * 60)
                logger.info("collect games %s ~ %s", start, end)
                try:
                    collect_games(start, end, client=client, team_codes=args.team_codes)
                except Exception as e:  # noqa: BLE001
                    logger.exception("month failed %s-%02d: %s", year, month, e)
                    # 월 단위 실패는 기록 후 계속 진행
                    continue

    games_path = config.PROCESSED_DIR / "games.csv"
    if not games_path.exists():
        logger.error("games.csv가 없습니다. 먼저 games 수집이 필요합니다.")
        return

    games_df = pd.read_csv(games_path, dtype={"game_id": str})

    # 수집 연도 범위로 필터 (이미 수집된 다른 연도는 건너뜀)
    year_prefix = games_df["game_id"].str[:4].astype(int)
    mask = (year_prefix >= args.start_year) & (year_prefix <= args.end_year)
    if args.team_codes:
        mask &= (
            games_df["away_team_code"].isin(args.team_codes)
            | games_df["home_team_code"].isin(args.team_codes)
        )
    target_games = games_df[mask].reset_index(drop=True)
    logger.info("target games: %d", len(target_games))

    # ----------------------------------------------------------
    # 2) record 수집/파싱
    # ----------------------------------------------------------
    if not args.skip_records:
        collect_records(games_df=target_games, client=client)

    # ----------------------------------------------------------
    # 3) relay raw 수집 + 선수 마스터
    # ----------------------------------------------------------
    if not args.skip_relay:
        collect_relays(games_df=target_games, client=client)

    logger.info("collection done. 다음 단계: scripts/run_build_features.py")


if __name__ == "__main__":
    main()
