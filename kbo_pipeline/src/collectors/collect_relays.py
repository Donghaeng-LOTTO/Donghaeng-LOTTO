"""relay API 일괄 수집.

산출물:
- data/raw/naver_relay/{game_id}/{game_id}_base.json
- data/raw/naver_relay/{game_id}/{game_id}_inning_NN.json
- data/processed/naver_players_seen.csv (lineup 기반 선수 마스터)
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src import config
from src.clients.naver_api import NaverClient, ensure_dirs, log_failed_request

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # noqa: D103
        return x


# ------------------------------------------------------------
# 정규화 유틸 (player map과 공유)
# ------------------------------------------------------------
def normalize_name(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip().replace(" ", "")


def normalize_birthdate(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    nums = re.findall(r"\d+", text)
    if len(nums) >= 3:
        y, m, d = nums[:3]
        return f"{int(y):04d}{int(m):02d}{int(d):02d}"
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return digits
    return None


def parse_naver_hit_type(hit_type):
    """예: 우투우타, 좌투좌타, 우언우타"""
    if not isinstance(hit_type, str):
        return None, None
    throw_hand = bat_hand = None
    if hit_type.startswith("우투"):
        throw_hand = "R"
    elif hit_type.startswith("좌투"):
        throw_hand = "L"
    elif hit_type.startswith("우언"):
        throw_hand = "R_submarine"
    elif hit_type.startswith("좌언"):
        throw_hand = "L_submarine"
    if hit_type.endswith("우타"):
        bat_hand = "R"
    elif hit_type.endswith("좌타"):
        bat_hand = "L"
    elif hit_type.endswith("양타"):
        bat_hand = "S"
    return throw_hand, bat_hand


# ------------------------------------------------------------
# lineup -> 선수 행
# ------------------------------------------------------------
def extract_players_from_lineup(lineup: dict, game_id: str, game_date,
                                team_side: str, team_code) -> list[dict]:
    rows = []
    for role in ["batter", "pitcher"]:
        for p in lineup.get(role, []) or []:
            naver_pcode = p.get("pcode") or p.get("playerCode")
            name = p.get("name")
            hit_type = p.get("hitType")
            throw_hand, bat_hand = parse_naver_hit_type(hit_type)
            rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "team_side": team_side,
                "team_code": team_code,
                "source_role": role,
                "naver_pcode": str(naver_pcode) if naver_pcode is not None else None,
                "naver_name": name,
                "name_norm": normalize_name(name),
                "birth_raw": p.get("birth"),
                "birthdate_norm": normalize_birthdate(p.get("birth")),
                "hit_type_raw": hit_type,
                "throw_hand": throw_hand,
                "bat_hand": bat_hand,
                "pos_name": p.get("posName"),
                "pos": p.get("pos"),
                "backnum": p.get("backnum"),
                "height": p.get("height"),
                "weight": p.get("weight"),
            })
    return rows


def extract_players_from_relay_json(data: dict, game_id: str, game_date,
                                    away_team_code=None, home_team_code=None) -> list[dict]:
    trd = data.get("result", {}).get("textRelayData", {}) or {}
    rows = []
    rows.extend(extract_players_from_lineup(
        trd.get("awayLineup", {}) or {}, game_id, game_date, "away", away_team_code))
    rows.extend(extract_players_from_lineup(
        trd.get("homeLineup", {}) or {}, game_id, game_date, "home", home_team_code))
    return rows


def extract_players_from_record_json(record_json: dict, game_id: str, game_date=None) -> list[dict]:
    """relay lineup이 비어 있을 때 record boxscore에서 pcode/name만이라도 수집."""
    rd = record_json.get("result", {}).get("recordData", {}) or {}
    gi = rd.get("gameInfo", {}) or {}
    rows = []
    for side in ["away", "home"]:
        team_code = gi.get("aCode") if side == "away" else gi.get("hCode")
        for b in (rd.get("battersBoxscore", {}) or {}).get(side, []) or []:
            pcode, name = b.get("playerCode"), b.get("name")
            if not pcode or not name:
                continue
            rows.append({
                "game_id": game_id, "game_date": game_date or gi.get("gdate"),
                "team_side": side, "team_code": team_code,
                "source_role": "batter_record",
                "naver_pcode": str(pcode), "naver_name": name,
                "name_norm": normalize_name(name),
                "birth_raw": None, "birthdate_norm": None,
                "hit_type_raw": None, "throw_hand": None, "bat_hand": None,
                "pos_name": b.get("pos"), "pos": b.get("pos"),
                "backnum": None, "height": None, "weight": None,
            })
        for p in (rd.get("pitchersBoxscore", {}) or {}).get(side, []) or []:
            pcode, name = p.get("pcode"), p.get("name")
            if not pcode or not name:
                continue
            rows.append({
                "game_id": game_id, "game_date": game_date or gi.get("gdate"),
                "team_side": side, "team_code": team_code,
                "source_role": "pitcher_record",
                "naver_pcode": str(pcode), "naver_name": name,
                "name_norm": normalize_name(name),
                "birth_raw": None, "birthdate_norm": None,
                "hit_type_raw": None, "throw_hand": None, "bat_hand": None,
                "pos_name": "투수", "pos": "P",
                "backnum": None, "height": None, "weight": None,
            })
    return rows


# ------------------------------------------------------------
# relay raw 수집 (base + 이닝별)
# ------------------------------------------------------------
def collect_relay_for_game(client: NaverClient, game_id: str, max_inning: int = 18) -> dict:
    """한 경기의 relay raw 수집. raw JSON 캐시 우선.

    returns: {"base_ok": bool, "innings": [int,...], "base_data": dict|None}
    """
    result = {"base_ok": False, "innings": [], "base_data": None}

    try:
        base_data, from_cache = client.fetch_relay_cached(game_id)
        result["base_ok"] = True
        result["base_data"] = base_data
        if not from_cache:
            client.polite_sleep()
    except Exception as e:  # noqa: BLE001
        logger.warning("[relay base failed] %s / %s", game_id, e)
        log_failed_request("relay_base", game_id, repr(e))

    empty_streak = 0
    for inning in range(1, max_inning + 1):
        cache_path = client.relay_cache_path(game_id, inning)
        cached = cache_path.exists()

        try:
            data, from_cache = client.fetch_relay_cached(game_id, inning)
        except Exception as e:  # noqa: BLE001
            logger.warning("[relay inning failed] %s inn=%d / %s", game_id, inning, e)
            log_failed_request("relay_inning", f"{game_id}:{inning}", repr(e))
            empty_streak += 1
            if empty_streak >= 2 and inning >= 9:
                break
            continue

        text_relays = (
            data.get("result", {}).get("textRelayData", {}) or {}
        ).get("textRelays", []) or []

        if not text_relays:
            # 빈 이닝: 캐시로 만들어진 빈 파일은 지워서 디스크 절약
            if not cached and cache_path.exists():
                cache_path.unlink(missing_ok=True)
            empty_streak += 1
            # 9회 이후 연속 2이닝 비어 있으면 연장 없음으로 보고 중단
            if empty_streak >= 2 and inning >= 9:
                break
        else:
            empty_streak = 0
            result["innings"].append(inning)

        if not from_cache:
            client.polite_sleep()

    return result


def collect_relays(
    games_df: pd.DataFrame | None = None,
    client: NaverClient | None = None,
    max_inning: int = 18,
    flush_every: int = 50,
) -> None:
    """games.csv 전체에 대해 relay raw 수집 + 선수 마스터 갱신."""
    ensure_dirs()
    client = client or NaverClient()

    if games_df is None:
        games_df = pd.read_csv(config.PROCESSED_DIR / "games.csv", dtype={"game_id": str})

    players_out = config.PROCESSED_DIR / "naver_players_seen.csv"
    player_rows: list[dict] = []

    def flush_players():
        nonlocal player_rows
        if not player_rows:
            return
        new_df = pd.DataFrame(player_rows)
        if players_out.exists():
            old_df = pd.read_csv(players_out, dtype={"naver_pcode": str, "game_id": str})
            merged = pd.concat([old_df, new_df], ignore_index=True)
        else:
            merged = new_df
        merged = (
            merged
            .dropna(subset=["naver_pcode", "naver_name"])
            .drop_duplicates(subset=[
                "naver_pcode", "naver_name", "birthdate_norm",
                "hit_type_raw", "source_role",
            ])
            .sort_values(["naver_name", "naver_pcode"])
            .reset_index(drop=True)
        )
        merged.to_csv(players_out, index=False, encoding="utf-8-sig")
        logger.info("[saved] %s rows=%d", players_out, len(merged))
        player_rows = []

    for i, (_, game) in enumerate(tqdm(games_df.iterrows(), total=len(games_df), desc="relays"), start=1):
        game_id = str(game["game_id"])
        game_date = game.get("game_date")

        res = collect_relay_for_game(client, game_id, max_inning=max_inning)

        # lineup 선수 수집 (base -> inning1 -> record 순 fallback)
        rows: list[dict] = []
        if res["base_data"] is not None:
            rows = extract_players_from_relay_json(
                res["base_data"], game_id, game_date,
                away_team_code=game.get("away_team_code"),
                home_team_code=game.get("home_team_code"),
            )
        if not rows and 1 in res["innings"]:
            try:
                data, _ = client.fetch_relay_cached(game_id, 1)
                rows = extract_players_from_relay_json(
                    data, game_id, game_date,
                    away_team_code=game.get("away_team_code"),
                    home_team_code=game.get("home_team_code"),
                )
            except Exception:  # noqa: BLE001
                pass
        if not rows:
            try:
                record_json, _ = client.fetch_record_cached(game_id)
                rows = extract_players_from_record_json(record_json, game_id, game_date)
            except Exception as e:  # noqa: BLE001
                log_failed_request("players", game_id, repr(e))

        player_rows.extend(rows)

        if i % flush_every == 0:
            flush_players()

    flush_players()
