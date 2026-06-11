"""relay raw JSON -> relay_events / plate_appearances 변환."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config
from src.clients.naver_api import load_json

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# event_type 매핑
# ------------------------------------------------------------
EVENT_CLASS_MAP = {
    0: "inning_start",
    1: "pitch",
    2: "player_change",
    8: "batter_start",
    13: "plate_appearance_result",
    23: "plate_appearance_result_scoring",
    14: "runner_movement",
    24: "runner_movement_scoring",
    99: "system",
}

EVENT_GROUP_MAP = {
    0: "inning_start",
    1: "pitch",
    2: "player_change",
    8: "batter_start",
    13: "plate_appearance_result",
    23: "plate_appearance_result",
    14: "runner_movement",
    24: "runner_movement",
    99: "system",
}


def map_event_class(event_type) -> str:
    try:
        return EVENT_CLASS_MAP.get(int(event_type), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def map_event_group(event_type) -> str:
    try:
        return EVENT_GROUP_MAP.get(int(event_type), "unknown")
    except (TypeError, ValueError):
        return "unknown"

def first_valid(*values):
    for v in values:
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            return v
    return None


def _half_code_from_text(value) -> str | None:
    """문자중계 제목/텍스트에서 초/말을 추론한다.

    프로젝트 내부 표준:
    - "0" = 초 = 원정 공격
    - "1" = 말 = 홈 공격

    일부 2008 relay JSON은 relay.homeOrAway가 초/말을 제대로
    표현하지 못하고 이닝 전체가 같은 값으로 들어오므로,
    "1회초 KIA 공격" 같은 inning_start 텍스트를 더 신뢰한다.
    """
    if value is None or pd.isna(value):
        return None

    s = str(value).strip()

    if "회초" in s:
        return "0"
    if "회말" in s:
        return "1"

    return None


def _normalize_home_or_away_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    s = str(value).strip()

    if s in {"0", "0.0", "top", "away", "AWAY"}:
        return "0"
    if s in {"1", "1.0", "bottom", "bot", "home", "HOME"}:
        return "1"

    return None


def infer_home_or_away_from_text(events_df: pd.DataFrame) -> pd.DataFrame:
    """event_type=0의 'n회초/말 ... 공격' 텍스트를 기준으로 초/말 보정.

    네이버 과거 relay는 relay.homeOrAway가 전체 이닝에서 같은 값으로
    찍히는 케이스가 있어, 점수 재구성에서 away 득점이 모두 home에
    합산되는 문제가 생긴다. 따라서 텍스트 기반 half marker를 forward-fill한다.
    """
    if events_df.empty:
        return events_df

    df = events_df.sort_values(["seqno"]).reset_index(drop=True).copy()

    raw = df["home_or_away"].apply(_normalize_home_or_away_value)

    marker_from_text = df.get(
        "text",
        pd.Series(index=df.index, dtype=object)
    ).apply(_half_code_from_text)

    marker_from_title = df.get(
        "relay_title",
        pd.Series(index=df.index, dtype=object)
    ).apply(_half_code_from_text)

    marker = marker_from_text.combine_first(marker_from_title)
    inferred = marker.ffill()

    df["home_or_away_raw"] = df["home_or_away"]
    df["home_or_away"] = inferred.combine_first(raw)

    df["home_or_away_infer_source"] = marker.notna().map({
        True: "inning_text",
        False: None,
    })
    df["home_or_away_infer_source"] = df["home_or_away_infer_source"].ffill()

    return df

# ------------------------------------------------------------
# events 추출
# ------------------------------------------------------------
def extract_events_from_relay_json(data: dict, game_id: str, source_file: str) -> list[dict]:
    rows = []
    trd = data.get("result", {}).get("textRelayData", {}) or {}

    for relay in trd.get("textRelays", []) or []:
        relay_no = relay.get("no")
        inning = relay.get("inn")
        home_or_away = relay.get("homeOrAway")
        relay_title = relay.get("title")
        title_style = relay.get("titleStyle")

        for opt in relay.get("textOptions", []) or []:
            state = opt.get("currentGameState", {}) or {}
            batter_record = opt.get("batterRecord", {}) or {}
            player_change = opt.get("playerChange", {}) or {}

            rows.append({
                "game_id": game_id,
                "source_file": source_file,

                "relay_no": relay_no,
                "seqno": opt.get("seqno"),
                "inning": inning,
                "home_or_away": home_or_away,
                "relay_title": relay_title,
                "title_style": title_style,

                "event_type": opt.get("type"),
                "text": opt.get("text"),

                "pitch_num": opt.get("pitchNum"),
                "pitch_result": opt.get("pitchResult"),
                "speed": opt.get("speed"),
                "pitch_stuff": opt.get("stuff"),

                "home_score": state.get("homeScore"),
                "away_score": state.get("awayScore"),
                "home_hit": state.get("homeHit"),
                "away_hit": state.get("awayHit"),
                "home_bb": state.get("homeBallFour"),
                "away_bb": state.get("awayBallFour"),
                "home_error": state.get("homeError"),
                "away_error": state.get("awayError"),

                "ball": state.get("ball"),
                "strike": state.get("strike"),
                "out": state.get("out"),
                "base1": state.get("base1"),
                "base2": state.get("base2"),
                "base3": state.get("base3"),

                "pitcher_pcode": state.get("pitcher"),
                "batter_pcode": state.get("batter"),

                "batter_name": batter_record.get("name"),
                "batter_pos_name": batter_record.get("posName"),
                "batter_order": batter_record.get("batOrder"),
                "batter_hit_type": batter_record.get("hitType"),
                "batter_season_hra": batter_record.get("seasonHra"),
                "batter_vs_hra": batter_record.get("vsHra"),
                "batter_pcode_from_record": batter_record.get("pcode"),

                "has_player_change": bool(player_change),
                "player_change_text": player_change.get("liveText"),
            })

    return rows


def build_events_df_for_game(game_id: str) -> pd.DataFrame:
    """저장된 raw relay JSON들에서 한 경기 events_df 생성."""
    raw_dir = config.RAW_RELAY_DIR / game_id

    all_rows: list[dict] = []
    inning_files = sorted(raw_dir.glob(f"{game_id}_inning_*.json"))

    # 이닝별 파일이 없으면 base 파일이라도 사용
    if not inning_files:
        base = raw_dir / f"{game_id}_base.json"
        if base.exists():
            inning_files = [base]

    for path in inning_files:
        try:
            data = load_json(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[events load failed] %s / %s", path, e)
            continue
        all_rows.extend(extract_events_from_relay_json(data, game_id, path.name))

    events_df = pd.DataFrame(all_rows)
    if events_df.empty:
        return events_df

    # events_df = (
    #     events_df
    #     .drop_duplicates(subset=["game_id", "seqno"])
    #     .sort_values(["seqno"])
    #     .reset_index(drop=True)
    # )

    # events_df["event_class"] = events_df["event_type"].apply(map_event_class)

    events_df = (
    events_df
    .drop_duplicates(subset=["game_id", "seqno"])
    .sort_values(["seqno"])
    .reset_index(drop=True)
)

    # relay.homeOrAway가 과거 경기에서 전체 이닝에 같은 값으로 들어오는 경우가 있어
    # "1회초/1회말" 텍스트 기반으로 초/말을 재추론한다.
    events_df = infer_home_or_away_from_text(events_df)

    events_df["event_class"] = events_df["event_type"].apply(map_event_class)

    events_df["event_group"] = events_df["event_type"].apply(map_event_group)
    events_df["is_scoring_related"] = events_df["event_type"].isin([23, 24])
    return events_df


# ------------------------------------------------------------
# 결과 텍스트 분류
# ------------------------------------------------------------
def classify_pa_result(text) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "unknown"
    if "홈런" in text:
        return "home_run"
    if "3루타" in text:
        return "triple"
    if "2루타" in text:
        return "double"
    if "1루타" in text or "내야안타" in text or "안타" in text:
        return "single"
    if "고의4구" in text or "고의 4구" in text:
        return "intentional_walk"
    if "볼넷" in text or "4구" in text:
        return "walk"
    if "몸에 맞" in text or "사구" in text:
        return "hit_by_pitch"
    if "삼진" in text:
        return "strikeout"
    if "희생번트" in text:
        return "sac_bunt"
    if "희생플라이" in text or "희생 플라이" in text:
        return "sac_fly"
    if "병살" in text:
        return "double_play"
    if "실책" in text:
        return "error"
    if "야수선택" in text:
        return "fielders_choice"
    if "아웃" in text or "플라이" in text or "땅볼" in text or "라인드라이브" in text or "파울플라이" in text:
        return "out"
    return "other"


def classify_runner_movement(text) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "none"
    labels = []
    if "홈인" in text:
        labels.append("score")
    if "도루" in text:
        labels.append("steal")
    if "진루" in text or "까지" in text:
        labels.append("advance")
    if "아웃" in text:
        labels.append("runner_out")
    if "폭투" in text:
        labels.append("wild_pitch")
    if "포일" in text:
        labels.append("passed_ball")
    if "보크" in text:
        labels.append("balk")
    if "실책" in text:
        labels.append("error")
    return ",".join(labels) if labels else "other"


# ------------------------------------------------------------
# plate appearance 그룹화
# ------------------------------------------------------------
def build_plate_appearances_for_game(events_df: pd.DataFrame) -> pd.DataFrame:
    """relay_no 단위 그룹 -> 타석 단위 행."""
    if events_df.empty:
        return pd.DataFrame()

    pa_groups: list[dict] = []

    for relay_no, group in events_df.groupby("relay_no"):
        group = group.sort_values("seqno")
        event_types = set(group["event_type"].dropna().astype(int).tolist())

        has_batter_start = 8 in event_types
        has_result = bool(event_types.intersection({13, 23}))
        if not (has_batter_start and has_result):
            continue

        first_row = group.iloc[0]
        batter_rows = group[group["event_type"] == 8]
        result_rows = group[group["event_type"].isin([13, 23])]
        runner_rows = group[group["event_type"].isin([14, 24])]
        pitch_rows = group[group["event_type"] == 1]
        change_rows = group[group["event_type"] == 2]

        batter_row = batter_rows.iloc[-1] if len(batter_rows) else first_row
        result_row = result_rows.iloc[0]
        result_seqno = result_row["seqno"]

        runner_before = runner_rows[runner_rows["seqno"] < result_seqno]
        runner_after = runner_rows[runner_rows["seqno"] > result_seqno]

        pa_groups.append({
            "game_id": first_row["game_id"],
            "relay_no": relay_no,
            "inning": first_row["inning"],
            "home_or_away": first_row["home_or_away"],

            "start_seqno": group["seqno"].min(),
            "result_seqno": result_seqno,
            "end_seqno": group["seqno"].max(),

            "batter_pcode": first_valid(
                batter_row.get("batter_pcode"),
                batter_row.get("batter_pcode_from_record"),
            ),
            "pitcher_pcode": batter_row.get("pitcher_pcode"),
            "batter_name": batter_row.get("batter_name"),
            "batter_order": batter_row.get("batter_order"),
            "batter_pos_name": batter_row.get("batter_pos_name"),

            "pitch_count": len(pitch_rows),
            "result_event_type": result_row["event_type"],
            "result_text": " | ".join(result_rows["text"].dropna().tolist()),

            "runner_before_result_text": " | ".join(runner_before["text"].dropna().tolist()),
            "runner_after_result_text": " | ".join(runner_after["text"].dropna().tolist()),
            "change_text": " | ".join(change_rows["text"].dropna().tolist()),

            "has_runner_movement_before_result": len(runner_before) > 0,
            "has_runner_movement_after_result": len(runner_after) > 0,
            "has_player_change": len(change_rows) > 0,
            "is_scoring_pa_result": int(result_row["event_type"] == 23),
        })

    pa_df = pd.DataFrame(pa_groups)
    if pa_df.empty:
        return pa_df

    pa_df["pa_result_type"] = pa_df["result_text"].apply(classify_pa_result)
    pa_df = pa_df.sort_values("start_seqno").reset_index(drop=True)
    return pa_df


# def attach_pitcher_names(pa_df: pd.DataFrame, players_df: pd.DataFrame | None) -> pd.DataFrame:
#     """naver_players_seen 기준으로 pitcher_name을 붙인다 (가능하면)."""
#     if pa_df.empty or players_df is None or players_df.empty:
#         pa_df["pitcher_name"] = None
#         return pa_df

#     name_map = (
#         players_df
#         .dropna(subset=["naver_pcode"])
#         .drop_duplicates(subset=["naver_pcode"])
#         .set_index(players_df["naver_pcode"].astype(str))["naver_name"]
#     )
#     pa_df["pitcher_name"] = pa_df["pitcher_pcode"].astype(str).map(name_map)
#     return pa_df

def clean_pcode(value):
    """
    pcode가 CSV를 거치면서 71851.0 같은 float 문자열이 되는 경우를 정리한다.
    """
    if value is None:
        return None

    s = str(value).strip()

    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    if s.endswith(".0"):
        s = s[:-2]

    return s


def build_player_name_map(players_df: pd.DataFrame) -> dict:
    if players_df is None or players_df.empty:
        return {}

    players = players_df.copy()

    if "naver_pcode" not in players.columns or "naver_name" not in players.columns:
        return {}

    players["naver_pcode_norm"] = players["naver_pcode"].apply(clean_pcode)
    players = players.dropna(subset=["naver_pcode_norm", "naver_name"])

    return (
        players[["naver_pcode_norm", "naver_name"]]
        .drop_duplicates(subset=["naver_pcode_norm"])
        .set_index("naver_pcode_norm")["naver_name"]
        .to_dict()
    )


def attach_pitcher_names(pa_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 함수명 유지.
    pitcher_name을 붙이고, 가능하면 batter_name 결측도 보강한다.
    """
    if pa_df is None or pa_df.empty:
        return pa_df

    pa_df = pa_df.copy()
    name_map = build_player_name_map(players_df)

    if "pitcher_pcode" in pa_df.columns:
        pa_df["pitcher_pcode_norm"] = pa_df["pitcher_pcode"].apply(clean_pcode)
        pa_df["pitcher_name"] = pa_df["pitcher_pcode_norm"].map(name_map)
    else:
        pa_df["pitcher_name"] = None

    if "batter_pcode" in pa_df.columns:
        pa_df["batter_pcode_norm"] = pa_df["batter_pcode"].apply(clean_pcode)

        if "batter_name" not in pa_df.columns:
            pa_df["batter_name"] = None

        pa_df["batter_name"] = pa_df["batter_name"].fillna(
            pa_df["batter_pcode_norm"].map(name_map)
        )

    return pa_df

# ------------------------------------------------------------
# event_type 코드북 (탐색용)
# ------------------------------------------------------------
def build_event_type_codebook(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    return (
        events_df
        .groupby("event_type")
        .agg(
            count=("text", "count"),
            examples=("text", lambda x: list(x.dropna().head(10))),
        )
        .reset_index()
        .sort_values("event_type")
    )
