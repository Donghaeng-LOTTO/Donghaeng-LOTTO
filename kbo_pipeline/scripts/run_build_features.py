"""수집된 raw에서 최종 데이터셋 생성.

순서:
1) relay raw -> relay_events.csv, plate_appearances.csv (+상태 재구성, 파생 컬럼)
2) score_validation.csv (재구성 점수 vs record 최종 점수)
3) player_id_map.csv (Kaggle 매핑)
4) batter/pitcher_pre_game_stats.csv (shift(1) 누수 방지)
5) dataset_quality_report.csv

사용 예:
  uv run python scripts/run_build_features.py
  uv run python scripts/run_build_features.py --skip-events   # events/PA 재생성 생략
  uv run python scripts/run_build_features.py --no-season-reset  # 통산 누적으로 pre-game 생성
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.parsers.parse_relays import (  # noqa: E402
    build_events_df_for_game,
    build_plate_appearances_for_game,
    attach_pitcher_names,
)
from src.parsers.state_reconstructor import (  # noqa: E402
    reconstruct_states_for_game,
    validate_final_score,
)
from src.features.pa_features import add_pa_derived_columns  # noqa: E402
from src.features.player_id_map import run_player_id_map  # noqa: E402
from src.features.pre_game_stats import run_pre_game_stats  # noqa: E402
from src.quality.quality_report import build_quality_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_build_features")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # noqa: D103
        return x


def build_events_and_pas() -> None:
    games_detail = pd.read_csv(
        config.PROCESSED_DIR / "games_detail.csv", dtype={"game_id": str})

    players_path = config.PROCESSED_DIR / "naver_players_seen.csv"
    players_df = (
        pd.read_csv(players_path, dtype={"naver_pcode": str})
        if players_path.exists() else None
    )

    events_out = config.PROCESSED_DIR / "relay_events.csv"
    pa_out = config.PROCESSED_DIR / "plate_appearances.csv"
    valid_out = config.PROCESSED_DIR / "score_validation.csv"

    # 대용량이므로 게임 단위로 append (헤더는 첫 게임만)
    for p in (events_out, pa_out, valid_out):
        p.unlink(missing_ok=True)

    first_events = True
    first_pa = True
    valid_rows = []

    game_ids = games_detail["game_id"].astype(str).tolist()

    for game_id in tqdm(game_ids, desc="parse relays"):
        events_df = build_events_df_for_game(game_id)
        if events_df.empty:
            continue

        events_df.to_csv(
            events_out, mode="a", header=first_events,
            index=False, encoding="utf-8-sig")
        first_events = False

        pa_df = build_plate_appearances_for_game(events_df)
        if pa_df.empty:
            continue

        pa_df = reconstruct_states_for_game(pa_df, events_df)
        pa_df = attach_pitcher_names(pa_df, players_df)
        pa_df = add_pa_derived_columns(pa_df, games_detail)

        pa_df.to_csv(
            pa_out, mode="a", header=first_pa,
            index=False, encoding="utf-8-sig")
        first_pa = False

        gd = games_detail[games_detail["game_id"] == game_id]
        away_r = gd["away_score"].iloc[0] if len(gd) else None
        home_r = gd["home_score"].iloc[0] if len(gd) else None
        v = validate_final_score(pa_df, away_r, home_r)
        v["game_id"] = game_id
        valid_rows.append(v)

    pd.DataFrame(valid_rows).to_csv(valid_out, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s / %s / %s", events_out, pa_out, valid_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-events", action="store_true",
                        help="relay_events/plate_appearances 재생성 생략")
    parser.add_argument("--skip-player-map", action="store_true")
    parser.add_argument("--skip-pre-game", action="store_true")
    parser.add_argument("--no-season-reset", action="store_true",
                        help="pre-game 누적을 시즌 리셋 없이 통산으로 계산")
    args = parser.parse_args()

    if not args.skip_events:
        build_events_and_pas()

    if not args.skip_player_map:
        try:
            run_player_id_map()
        except FileNotFoundError as e:
            logger.warning("player map 생략 (Kaggle CSV 또는 players_seen 없음): %s", e)

    if not args.skip_pre_game:
        run_pre_game_stats(cumulate_within_season=not args.no_season_reset)

    build_quality_report()
    logger.info("done.")


if __name__ == "__main__":
    main()
