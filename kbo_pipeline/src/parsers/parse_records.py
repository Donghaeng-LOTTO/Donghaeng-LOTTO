"""record JSON -> games_detail / batter_game_boxscores / pitcher_game_boxscores 행 추출."""
from __future__ import annotations


def parse_innings_to_outs(value) -> int | None:
    """'7 ⅓', '0 ⅔', '1', '7 1/3', '5.1' 같은 이닝 표기를 아웃카운트로 변환."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None

    if "⅓" in s:
        whole = s.replace("⅓", "").strip()
        return (int(whole) * 3 if whole else 0) + 1
    if "⅔" in s:
        whole = s.replace("⅔", "").strip()
        return (int(whole) * 3 if whole else 0) + 2
    if "1/3" in s:
        whole = s.replace("1/3", "").strip()
        return (int(whole) * 3 if whole else 0) + 1
    if "2/3" in s:
        whole = s.replace("2/3", "").strip()
        return (int(whole) * 3 if whole else 0) + 2
    if "." in s:
        whole, frac = s.split(".", 1)
        return int(whole) * 3 + int(frac)

    return int(s) * 3


def extract_game_detail_row(record_json: dict, game_id: str) -> dict:
    rd = record_json["result"]["recordData"]
    gi = rd["gameInfo"]
    sb = rd.get("scoreBoard", {}) or {}

    rheb = sb.get("rheb", {}) or {}
    away_score = (rheb.get("away") or {}).get("r")
    home_score = (rheb.get("home") or {}).get("r")

    if away_score is None or home_score is None:
        winner = None
    elif away_score > home_score:
        winner = gi.get("aCode")
    elif home_score > away_score:
        winner = gi.get("hCode")
    else:
        winner = "DRAW"

    return {
        "game_id": game_id,
        "game_date": gi.get("gdate"),
        "game_time": gi.get("gtime"),
        "stadium": gi.get("stadium"),

        "away_team_code": gi.get("aCode"),
        "away_team_name": gi.get("aName"),
        "away_team_full_name": gi.get("aFullName"),
        "home_team_code": gi.get("hCode"),
        "home_team_name": gi.get("hName"),
        "home_team_full_name": gi.get("hFullName"),

        "away_starting_pitcher_pcode": gi.get("aPCode"),
        "home_starting_pitcher_pcode": gi.get("hPCode"),

        "away_score": away_score,
        "home_score": home_score,
        "winner_team_code": winner,

        "status_code": gi.get("statusCode"),
        "cancel_flag": gi.get("cancelFlag"),
    }


def extract_inning_scores(record_json: dict, game_id: str) -> dict:
    """scoreBoard의 이닝별 점수 (검증용)."""
    rd = record_json["result"]["recordData"]
    sb = rd.get("scoreBoard", {}) or {}
    row = {"game_id": game_id}
    for side in ["away", "home"]:
        inn_scores = sb.get("inn", {}) or {}
        scores = inn_scores.get(side) or []
        for i, s in enumerate(scores, start=1):
            row[f"{side}_inn{i}"] = s
        rheb = (sb.get("rheb", {}) or {}).get(side) or {}
        row[f"{side}_r"] = rheb.get("r")
        row[f"{side}_h"] = rheb.get("h")
        row[f"{side}_e"] = rheb.get("e")
        row[f"{side}_b"] = rheb.get("b")
    return row


def extract_batter_boxscores(record_json: dict, game_id: str) -> list[dict]:
    rd = record_json["result"]["recordData"]
    gi = rd["gameInfo"]

    rows = []
    for side in ["away", "home"]:
        team_code = gi.get("aCode") if side == "away" else gi.get("hCode")
        team_name = gi.get("aName") if side == "away" else gi.get("hName")
        batters = (rd.get("battersBoxscore", {}) or {}).get(side, []) or []

        for b in batters:
            row = {
                "game_id": game_id,
                "game_date": gi.get("gdate"),
                "team_side": side,
                "team_code": team_code,
                "team_name": team_name,

                "player_code": b.get("playerCode"),
                "name": b.get("name"),
                "bat_order": b.get("batOrder"),
                "pos": b.get("pos"),

                "ab": b.get("ab"),
                "hit": b.get("hit"),
                "hr": b.get("hr"),
                "bb": b.get("bb"),
                "kk": b.get("kk"),
                "rbi": b.get("rbi"),
                "run": b.get("run"),
                "hra": b.get("hra"),
                "has_player_end": b.get("hasPlayerEnd"),
            }
            for i in range(1, 26):
                row[f"inn{i}"] = b.get(f"inn{i}")
            rows.append(row)

    return rows


def extract_pitcher_boxscores(record_json: dict, game_id: str) -> list[dict]:
    rd = record_json["result"]["recordData"]
    gi = rd["gameInfo"]

    rows = []
    for side in ["away", "home"]:
        team_code = gi.get("aCode") if side == "away" else gi.get("hCode")
        team_name = gi.get("aName") if side == "away" else gi.get("hName")
        pitchers = (rd.get("pitchersBoxscore", {}) or {}).get(side, []) or []

        for p in pitchers:
            inn = p.get("inn")
            outs = parse_innings_to_outs(inn)
            rows.append({
                "game_id": game_id,
                "game_date": gi.get("gdate"),
                "team_side": side,
                "team_code": team_code,
                "team_name": team_name,

                "pcode": p.get("pcode"),
                "name": p.get("name"),

                "inn_raw": inn,
                "outs": outs,
                "ip": outs / 3 if outs is not None else None,

                "pa": p.get("pa"),
                "bf": p.get("bf"),
                "ab": p.get("ab"),

                "hit": p.get("hit"),
                "hr": p.get("hr"),
                "bb": p.get("bb"),
                "bbhp": p.get("bbhp"),
                "kk": p.get("kk"),
                "r": p.get("r"),
                "er": p.get("er"),

                "w": p.get("w"),
                "l": p.get("l"),
                "s": p.get("s"),
                "era": p.get("era"),
                "wls": p.get("wls"),
                "has_player_end": p.get("hasPlayerEnd"),
            })

    return rows
