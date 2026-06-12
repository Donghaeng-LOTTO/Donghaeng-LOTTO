"""네트워크 없이 파싱/피처 경로를 검증하는 합성 데이터 테스트."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import pandas as pd

from src import config
from src.parsers.parse_relays import (
    build_events_df_for_game, build_plate_appearances_for_game)
from src.parsers.state_reconstructor import (
    reconstruct_states_for_game, validate_final_score)
from src.features.pa_features import add_pa_derived_columns
from src.features.pre_game_stats import (
    build_batter_pre_game_stats, build_pitcher_pre_game_stats)

GAME_ID = "20100601HTLT0"


def opt(seqno, etype, text, state=None, batter=None):
    o = {"seqno": seqno, "type": etype, "text": text,
         "currentGameState": state or {}}
    if batter:
        o["batterRecord"] = batter
    return o


def make_relay_inning1():
    # 1회초(HT 공격): 단타 -> 도루 -> 홈런(2점) -> 삼진 x2... 간단히
    relays = [
        {"no": 1, "inn": 1, "homeOrAway": "0", "title": "1회초", "textOptions": [
            opt(1, 0, "1회초 HT 공격"),
        ]},
        {"no": 2, "inn": 1, "homeOrAway": "0", "title": "타자1", "textOptions": [
            opt(2, 8, "1번타자 김선두", batter={"name": "김선두", "batOrder": 1, "pcode": "B001"}),
            opt(3, 1, "1구 스트라이크"),
            opt(4, 13, "김선두 : 중전 안타 (1루타)"),
        ]},
        {"no": 3, "inn": 1, "homeOrAway": "0", "title": "타자2", "textOptions": [
            opt(5, 8, "2번타자 박둘째", batter={"name": "박둘째", "batOrder": 2, "pcode": "B002"}),
            opt(6, 14, "1루주자 김선두 : 도루 성공, 2루까지 진루"),
            opt(7, 1, "1구 볼"),
            opt(8, 23, "박둘째 : 좌월 홈런"),
            opt(9, 24, "2루주자 김선두 : 홈인"),
        ]},
        {"no": 4, "inn": 1, "homeOrAway": "0", "title": "타자3", "textOptions": [
            opt(10, 8, "3번타자 최삼번", batter={"name": "최삼번", "batOrder": 3, "pcode": "B003"}),
            opt(11, 13, "최삼번 : 헛스윙 삼진 아웃"),
        ]},
        {"no": 5, "inn": 1, "homeOrAway": "0", "title": "타자4", "textOptions": [
            opt(12, 8, "4번타자 정사번", batter={"name": "정사번", "batOrder": 4, "pcode": "B004"}),
            opt(13, 13, "정사번 : 유격수 땅볼 아웃"),
        ]},
        {"no": 6, "inn": 1, "homeOrAway": "0", "title": "타자5", "textOptions": [
            opt(14, 8, "5번타자 한오번", batter={"name": "한오번", "batOrder": 5, "pcode": "B005"}),
            opt(15, 13, "한오번 : 중견수 플라이 아웃"),
        ]},
        # 1회말(LT 공격): 볼넷 -> 병살 -> 아웃
        {"no": 7, "inn": 1, "homeOrAway": "1", "title": "1회말", "textOptions": [
            opt(16, 0, "1회말 LT 공격"),
        ]},
        {"no": 8, "inn": 1, "homeOrAway": "1", "title": "타자1", "textOptions": [
            opt(17, 8, "1번타자 이로테", batter={"name": "이로테", "batOrder": 1, "pcode": "B101"}),
            opt(18, 13, "이로테 : 볼넷 출루"),
        ]},
        {"no": 9, "inn": 1, "homeOrAway": "1", "title": "타자2", "textOptions": [
            opt(19, 8, "2번타자 김자이", batter={"name": "김자이", "batOrder": 2, "pcode": "B102"}),
            opt(20, 13, "김자이 : 2루수 앞 병살타"),
            opt(21, 14, "1루주자 이로테 : 2루에서 아웃"),
        ]},
        {"no": 10, "inn": 1, "homeOrAway": "1", "title": "타자3", "textOptions": [
            opt(22, 8, "3번타자 박부산", batter={"name": "박부산", "batOrder": 3, "pcode": "B103"}),
            opt(23, 13, "박부산 : 1루수 파울플라이 아웃"),
        ]},
    ]
    return {"result": {"textRelayData": {"gameId": GAME_ID, "textRelays": relays}}}


def main():
    raw_dir = config.RAW_RELAY_DIR / GAME_ID
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{GAME_ID}_inning_01.json").write_text(
        json.dumps(make_relay_inning1(), ensure_ascii=False), encoding="utf-8")

    events_df = build_events_df_for_game(GAME_ID)
    print("events:", events_df.shape)
    assert len(events_df) == 23, len(events_df)
    assert set(events_df["event_class"]) >= {"pitch", "batter_start"}

    pa_df = build_plate_appearances_for_game(events_df)
    print("PAs:", pa_df.shape)
    assert len(pa_df) == 8, len(pa_df)
    print(pa_df[["relay_no", "batter_name", "pa_result_type"]])
    assert list(pa_df["pa_result_type"]) == [
        "single", "home_run", "strikeout", "out", "out",
        "walk", "double_play", "out"]

    pa_df = reconstruct_states_for_game(pa_df, events_df)
    print(pa_df[["batter_name", "outs_before", "base_state_before",
                 "away_score_before", "home_score_before",
                 "state_source", "parse_warning"]])

    # 검증: 홈런 PA(두 번째) 직전 상태 = 0아웃, 주자 2루(도루는 결과 전 텍스트)
    hr = pa_df.iloc[1]
    assert hr["outs_before"] == 0
    assert hr["base_state_before"] == "100", hr["base_state_before"]  # 도루는 PA 도중 발생
    # 삼진 PA 직전: 2점 들어온 상태
    so = pa_df.iloc[2]
    assert so["away_score_before"] == 2, so["away_score_before"]
    # 1회말 병살 후 다음 타자 직전: 2아웃
    last = pa_df.iloc[7]
    assert last["outs_before"] == 2, last["outs_before"]

    # 파생 컬럼
    games_detail = pd.DataFrame([{
        "game_id": GAME_ID, "away_team_code": "HT", "home_team_code": "LT",
        "away_score": 2, "home_score": 0, "winner_team_code": "HT",
    }])
    pa_df = add_pa_derived_columns(pa_df, games_detail)
    print(pa_df[["batter_name", "is_top", "batting_team_code",
                 "is_lotte_batting", "score_diff_lotte_before",
                 "late_clutch", "final_win_label_lotte"]])
    assert pa_df.iloc[0]["is_top"] == True  # noqa: E712
    assert pa_df.iloc[0]["batting_team_code"] == "HT"
    assert pa_df.iloc[5]["is_lotte_batting"] == True  # noqa: E712
    assert pa_df.iloc[5]["final_win_label_lotte"] == 0.0
    assert pa_df.iloc[5]["score_diff_lotte_before"] == -2

    v = validate_final_score(pa_df, 2, 0)
    print("score validation:", v)

    # pre-game stats (shift(1) 누수 검증)
    box = pd.DataFrame([
        {"game_id": "G1", "game_date": "20100601", "team_side": "away",
         "team_code": "HT", "player_code": "B001", "name": "김선두",
         "ab": 4, "hit": 2, "hr": 1, "bb": 0, "kk": 1, "rbi": 2, "run": 1},
        {"game_id": "G2", "game_date": "20100602", "team_side": "away",
         "team_code": "HT", "player_code": "B001", "name": "김선두",
         "ab": 3, "hit": 1, "hr": 0, "bb": 1, "kk": 0, "rbi": 0, "run": 0},
        {"game_id": "G3", "game_date": "20100603", "team_side": "away",
         "team_code": "HT", "player_code": "B001", "name": "김선두",
         "ab": 4, "hit": 0, "hr": 0, "bb": 0, "kk": 2, "rbi": 0, "run": 0},
    ])
    b = build_batter_pre_game_stats(box)
    print(b[["game_id", "games_before", "cum_ab", "cum_hit", "avg_before"]])
    assert b.iloc[0]["cum_ab"] == 0          # 첫 경기 전 누적 0 (누수 없음)
    assert b.iloc[1]["cum_ab"] == 4 and b.iloc[1]["cum_hit"] == 2
    assert b.iloc[2]["cum_ab"] == 7 and b.iloc[2]["cum_hit"] == 3
    assert abs(b.iloc[2]["avg_before"] - 3 / 7) < 1e-9

    pbox = pd.DataFrame([
        {"game_id": "G1", "game_date": "20100601", "team_side": "home",
         "team_code": "LT", "pcode": "P001", "name": "송투수",
         "outs": 18, "r": 3, "er": 2, "hit": 5, "hr": 1, "bb": 2, "bbhp": 2,
         "kk": 7, "bf": 24},
        {"game_id": "G2", "game_date": "20100607", "team_side": "home",
         "team_code": "LT", "pcode": "P001", "name": "송투수",
         "outs": 21, "r": 1, "er": 1, "hit": 4, "hr": 0, "bb": 1, "bbhp": 1,
         "kk": 8, "bf": 26},
    ])
    p = build_pitcher_pre_game_stats(pbox)
    print(p[["game_id", "games_before", "ip_before", "era_before", "whip_before"]])
    assert p.iloc[0]["cum_outs"] == 0
    assert p.iloc[1]["cum_outs"] == 18
    assert abs(p.iloc[1]["era_before"] - (2 * 27 / 18)) < 1e-9   # 3.00
    assert abs(p.iloc[1]["whip_before"] - (2 + 5) / 6.0) < 1e-9

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
