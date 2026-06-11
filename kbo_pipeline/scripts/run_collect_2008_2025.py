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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from pandas.errors import EmptyDataError  # noqa: E402

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


# ------------------------------------------------------------
# 날짜 유틸
# ------------------------------------------------------------
def month_range(year: int, month: int) -> tuple[str, str]:
    """해당 연/월의 시작일, 마지막일을 YYYYMMDD 문자열로 반환."""
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}01", f"{year}{month:02d}{last_day:02d}"


def season_months(year: int, months: list[int] | None) -> list[int]:
    """수집할 월 목록 반환.

    months가 지정되면 그 월만 사용.
    지정되지 않으면 config.SEASON_START_MMDD ~ config.SEASON_END_MMDD 기준 사용.
    """
    if months:
        return months

    start_m = int(config.SEASON_START_MMDD[:2])
    end_m = int(config.SEASON_END_MMDD[:2])

    return list(range(start_m, end_m + 1))


def validate_months(months: list[int] | None) -> None:
    """월 입력값 검증."""
    if not months:
        return

    bad_months = [m for m in months if m < 1 or m > 12]
    if bad_months:
        raise ValueError(f"월은 1~12 사이여야 합니다. 잘못된 값: {bad_months}")


# ------------------------------------------------------------
# games.csv 안전 로드
# ------------------------------------------------------------
def load_games_csv(games_path: Path) -> pd.DataFrame | None:
    """games.csv를 안전하게 읽는다.

    - 파일 없음
    - 완전 빈 파일
    - 컬럼 없음
    - game_id 컬럼 없음
    - row 0개

    위 상황이면 None 반환.
    """
    if not games_path.exists():
        logger.error("games.csv가 없습니다: %s", games_path)
        logger.error("먼저 games 수집이 필요합니다.")
        return None

    try:
        games_df = pd.read_csv(games_path, dtype={"game_id": str})
    except EmptyDataError:
        logger.error("games.csv가 완전히 비어 있습니다: %s", games_path)
        logger.error("경기 목록 수집이 0건이라 이후 수집을 중단합니다.")
        return None
    except Exception as e:  # noqa: BLE001
        logger.exception("games.csv 읽기 실패: %s", e)
        return None

    if games_df.empty:
        logger.error("games.csv는 존재하지만 row가 0개입니다: %s", games_path)
        logger.error("경기 목록 수집이 0건이라 이후 수집을 중단합니다.")
        return None

    if "game_id" not in games_df.columns:
        logger.error("games.csv에 game_id 컬럼이 없습니다.")
        logger.error("현재 컬럼: %s", list(games_df.columns))
        return None

    games_df["game_id"] = games_df["game_id"].astype(str)

    return games_df


def filter_target_games(
    games_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    team_codes: list[str] | None = None,
) -> pd.DataFrame:
    """수집 대상 경기만 필터링."""

    # game_id 앞 4자리를 연도로 사용
    year_series = pd.to_numeric(games_df["game_id"].str[:4], errors="coerce")

    mask = (year_series >= start_year) & (year_series <= end_year)

    if team_codes:
        required_cols = {"away_team_code", "home_team_code"}
        missing = required_cols - set(games_df.columns)

        if missing:
            logger.warning(
                "팀 필터를 적용하려 했지만 필요한 컬럼이 없습니다: %s",
                sorted(missing),
            )
        else:
            mask &= (
                games_df["away_team_code"].isin(team_codes)
                | games_df["home_team_code"].isin(team_codes)
            )

    target_games = games_df[mask].copy().reset_index(drop=True)

    return target_games


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="특정 월만 수집. 예: --months 6 7",
    )
    parser.add_argument("--sleep-sec", type=float, default=config.DEFAULT_SLEEP_SEC)
    parser.add_argument(
        "--team-codes",
        type=str,
        nargs="*",
        default=None,
        help="특정 팀 경기만 수집. 예: --team-codes LT",
    )
    parser.add_argument("--skip-games", action="store_true")
    parser.add_argument("--skip-records", action="store_true")
    parser.add_argument("--skip-relay", action="store_true")

    args = parser.parse_args()

    if args.start_year > args.end_year:
        logger.error("--start-year가 --end-year보다 클 수 없습니다.")
        return

    try:
        validate_months(args.months)
    except ValueError as e:
        logger.error("%s", e)
        return

    ensure_dirs()

    client = NaverClient(sleep_sec=args.sleep_sec)

    # --------------------------------------------------------
    # 1) games.csv : 연/월 단위 수집
    # --------------------------------------------------------
    if not args.skip_games:
        for year in range(args.start_year, args.end_year + 1):
            months = season_months(year, args.months)

            for month in months:
                start, end = month_range(year, month)

                logger.info("=" * 60)
                logger.info("collect games %s ~ %s", start, end)

                try:
                    collect_games(
                        start,
                        end,
                        client=client,
                        team_codes=args.team_codes,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception("month failed %s-%02d: %s", year, month, e)
                    # 월 단위 실패는 기록 후 계속 진행
                    continue

    # --------------------------------------------------------
    # games.csv 안전 로드
    # --------------------------------------------------------
    games_path = config.PROCESSED_DIR / "games.csv"
    games_df = load_games_csv(games_path)

    if games_df is None:
        logger.error("중단합니다. collect_games 쪽에서 valid=0 원인을 먼저 확인해야 합니다.")
        return

    logger.info("loaded games.csv rows: %d", len(games_df))

    # --------------------------------------------------------
    # 수집 연도 / 팀 필터
    # --------------------------------------------------------
    target_games = filter_target_games(
        games_df=games_df,
        start_year=args.start_year,
        end_year=args.end_year,
        team_codes=args.team_codes,
    )

    logger.info("target games: %d", len(target_games))

    if target_games.empty:
        logger.error("대상 경기가 0건입니다.")
        logger.error("records / relay 수집을 건너뜁니다.")
        logger.error("가능한 원인:")
        logger.error("1) games.csv가 비어 있음")
        logger.error("2) game_id 형식이 예상과 다름")
        logger.error("3) collect_games 필터가 정상 경기를 전부 버림")
        logger.error("4) team_codes 필터가 너무 좁음")
        return

    # --------------------------------------------------------
    # 2) record 수집 / 파싱
    # --------------------------------------------------------
    if not args.skip_records:
        logger.info("=" * 60)
        logger.info("collect records")
        collect_records(games_df=target_games, client=client)
    else:
        logger.info("skip records")

    # --------------------------------------------------------
    # 3) relay raw 수집 + 선수 마스터
    # --------------------------------------------------------
    if not args.skip_relay:
        logger.info("=" * 60)
        logger.info("collect relays")
        collect_relays(games_df=target_games, client=client)
    else:
        logger.info("skip relay")

    logger.info("=" * 60)
    logger.info("collection done. 다음 단계: scripts/run_build_features.py")


if __name__ == "__main__":
    main()