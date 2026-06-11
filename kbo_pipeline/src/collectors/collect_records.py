"""games.csv 기준으로 record API 일괄 수집.

산출물:
- data/raw/naver_record/{game_id}/{game_id}_record.json  (raw 우선 저장)
- data/processed/games_detail.csv
- data/processed/batter_game_boxscores.csv
- data/processed/pitcher_game_boxscores.csv
- data/processed/scoreboard_inning_scores.csv (검증용)
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config
from src.clients.naver_api import NaverClient, ensure_dirs, log_failed_request
from src.parsers.parse_records import (
    extract_game_detail_row,
    extract_batter_boxscores,
    extract_pitcher_boxscores,
    extract_inning_scores,
)

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # tqdm은 선택
    def tqdm(x, **kwargs):  # noqa: D103
        return x


def _merge_save(new_rows: list[dict], out_name: str, key_cols: list[str]) -> pd.DataFrame:
    out_path = config.PROCESSED_DIR / out_name
    new_df = pd.DataFrame(new_rows)

    if out_path.exists():
        old_df = pd.read_csv(out_path, dtype={"game_id": str})
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    if not merged.empty:
        merged["game_id"] = merged["game_id"].astype(str)
        merged = (
            merged
            .drop_duplicates(subset=key_cols, keep="last")
            .sort_values(key_cols)
            .reset_index(drop=True)
        )

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s rows=%d", out_path, len(merged))
    return merged


def collect_records(
    games_df: pd.DataFrame | None = None,
    client: NaverClient | None = None,
    flush_every: int = 100,
) -> None:
    """games_df의 모든 game_id에 대해 record 수집 + 파싱.

    flush_every 경기마다 중간 저장하여 대량 수집 중단에도 결과가 보존되게 한다.
    """
    ensure_dirs()
    client = client or NaverClient()

    if games_df is None:
        games_df = pd.read_csv(config.PROCESSED_DIR / "games.csv", dtype={"game_id": str})

    game_ids = games_df["game_id"].astype(str).tolist()

    detail_rows: list[dict] = []
    batter_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    inning_rows: list[dict] = []

    def flush():
        if detail_rows:
            _merge_save(detail_rows, "games_detail.csv", ["game_id"])
            detail_rows.clear()
        if batter_rows:
            _merge_save(batter_rows, "batter_game_boxscores.csv",
                        ["game_id", "team_side", "player_code"])
            batter_rows.clear()
        if pitcher_rows:
            _merge_save(pitcher_rows, "pitcher_game_boxscores.csv",
                        ["game_id", "team_side", "pcode"])
            pitcher_rows.clear()
        if inning_rows:
            _merge_save(inning_rows, "scoreboard_inning_scores.csv", ["game_id"])
            inning_rows.clear()

    for i, game_id in enumerate(tqdm(game_ids, desc="records"), start=1):
        try:
            record_json, from_cache = client.fetch_record_cached(game_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[record failed] %s / %s", game_id, e)
            log_failed_request("record", game_id, repr(e))
            continue

        try:
            detail_rows.append(extract_game_detail_row(record_json, game_id))
            batter_rows.extend(extract_batter_boxscores(record_json, game_id))
            pitcher_rows.extend(extract_pitcher_boxscores(record_json, game_id))
            inning_rows.append(extract_inning_scores(record_json, game_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("[record parse failed] %s / %s", game_id, e)
            log_failed_request("record_parse", game_id, repr(e))

        if not from_cache:
            client.polite_sleep()

        if i % flush_every == 0:
            flush()

    flush()
