"""네이버 pcode <-> Kaggle Id 매핑.

1순위: name_norm + birthdate_norm  (confidence 1.0)
2순위: name_norm only, 후보가 정확히 1명일 때만 low confidence (0.5)

match_status / match_method / match_confidence / name_candidate_count 를 남긴다.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config
from src.collectors.collect_relays import normalize_name, normalize_birthdate

logger = logging.getLogger(__name__)


def normalize_kaggle_handedness(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def load_kaggle_player_identities() -> pd.DataFrame:
    batting = pd.read_csv(config.KAGGLE_BATTING_BY_SEASON)
    pitching = pd.read_csv(config.KAGGLE_PITCHING_BY_SEASON)

    batting_id = batting[["Id", "Name", "Birthdate", "Handedness"]].copy()
    batting_id["kaggle_source_role"] = "batter"
    pitching_id = pitching[["Id", "Name", "Birthdate", "Handedness"]].copy()
    pitching_id["kaggle_source_role"] = "pitcher"

    kaggle = pd.concat([batting_id, pitching_id], ignore_index=True)
    kaggle["kaggle_id"] = kaggle["Id"].astype(str)
    kaggle["kaggle_name"] = kaggle["Name"]
    kaggle["name_norm"] = kaggle["Name"].apply(normalize_name)
    kaggle["birthdate_norm"] = kaggle["Birthdate"].apply(normalize_birthdate)
    kaggle["kaggle_handedness_raw"] = kaggle["Handedness"].apply(normalize_kaggle_handedness)

    kaggle = kaggle[[
        "kaggle_id", "kaggle_name", "name_norm", "birthdate_norm",
        "kaggle_handedness_raw", "kaggle_source_role",
    ]]

    # 같은 선수가 타자/투수 양쪽에 있으면 role을 합쳐 한 줄로
    kaggle = (
        kaggle
        .groupby(
            ["kaggle_id", "kaggle_name", "name_norm",
             "birthdate_norm", "kaggle_handedness_raw"],
            dropna=False,
        )["kaggle_source_role"]
        .agg(lambda x: ",".join(sorted(set(x))))
        .reset_index()
    )
    return kaggle

def normalize_merge_key(value):
    """
    merge key용 문자열 정규화.
    CSV를 거치면서 19850818이 int/float로 읽히는 문제를 방지한다.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    # 19850818.0 -> 19850818
    if s.endswith(".0"):
        s = s[:-2]

    return s


def build_player_id_map(naver_players_df: pd.DataFrame, kaggle_players_df: pd.DataFrame) -> pd.DataFrame:
    naver_players_df = naver_players_df.copy()
    kaggle_players_df = kaggle_players_df.copy()

    # merge key 타입 통일
    for df in [naver_players_df, kaggle_players_df]:
        if "name_norm" in df.columns:
            df["name_norm"] = df["name_norm"].apply(normalize_merge_key)

        if "birthdate_norm" in df.columns:
            df["birthdate_norm"] = df["birthdate_norm"].apply(normalize_merge_key)

    if "naver_pcode" in naver_players_df.columns:
        naver_players_df["naver_pcode"] = naver_players_df["naver_pcode"].apply(normalize_merge_key)

    if "kaggle_id" in kaggle_players_df.columns:
        kaggle_players_df["kaggle_id"] = kaggle_players_df["kaggle_id"].apply(normalize_merge_key)

    naver_unique = (
        naver_players_df
        .sort_values(["naver_name", "naver_pcode"])
        .drop_duplicates(subset=["naver_pcode"])
        .copy()
    )

    merged = naver_unique.merge(
        kaggle_players_df,
        on=["name_norm", "birthdate_norm"],
        how="left",
        suffixes=("_naver", "_kaggle")
    )
# def build_player_id_map(
#     naver_players_df: pd.DataFrame, kaggle_players_df: pd.DataFrame
# ) -> pd.DataFrame:
#     naver_unique = (
#         naver_players_df
#         .sort_values(["naver_name", "naver_pcode"])
#         .drop_duplicates(subset=["naver_pcode"])
#         .copy()
#     )

#     # 1차: name + birthdate
#     merged = naver_unique.merge(
#         kaggle_players_df,
#         on=["name_norm", "birthdate_norm"],
#         how="left",
#         suffixes=("_naver", "_kaggle"),
#     )
    merged["match_method"] = None
    merged["match_confidence"] = 0.0
    exact_mask = merged["kaggle_id"].notna()
    merged.loc[exact_mask, "match_method"] = "name_birthdate"
    merged.loc[exact_mask, "match_confidence"] = 1.0

    # 2차: name only, 후보 1명
    unmatched = merged[merged["kaggle_id"].isna()].copy()

    name_candidate_count = (
        kaggle_players_df
        .groupby("name_norm")["kaggle_id"]
        .nunique()
        .reset_index(name="name_candidate_count")
    )
    single_names = name_candidate_count[
        name_candidate_count["name_candidate_count"] == 1
    ][["name_norm"]]
    kaggle_single = kaggle_players_df.merge(single_names, on="name_norm", how="inner")

    fallback = unmatched.drop(
        columns=[
            "kaggle_id", "kaggle_name", "kaggle_handedness_raw",
            "kaggle_source_role", "match_method", "match_confidence",
        ],
        errors="ignore",
    ).merge(kaggle_single, on="name_norm", how="left", suffixes=("", "_fallback"))

    fallback["match_method"] = fallback["kaggle_id"].notna().map(
        {True: "name_only_unique", False: None})
    fallback["match_confidence"] = fallback["kaggle_id"].notna().map(
        {True: 0.5, False: 0.0})

    final_map = pd.concat(
        [merged[merged["kaggle_id"].notna()], fallback], ignore_index=True)

    final_map = final_map.merge(name_candidate_count, on="name_norm", how="left")
    final_map["match_status"] = final_map["kaggle_id"].notna().map(
        {True: "matched", False: "unmatched"})

    keep_cols = [
        "naver_pcode", "naver_name", "birth_raw", "birthdate_norm",
        "hit_type_raw", "throw_hand", "bat_hand", "pos_name", "source_role",
        "kaggle_id", "kaggle_name", "kaggle_handedness_raw", "kaggle_source_role",
        "match_status", "match_method", "match_confidence", "name_candidate_count",
    ]
    for col in keep_cols:
        if col not in final_map.columns:
            final_map[col] = None

    return (
        final_map[keep_cols]
        .sort_values(["match_status", "naver_name", "naver_pcode"])
        .reset_index(drop=True)
    )


def run_player_id_map() -> pd.DataFrame:
    players_path = config.PROCESSED_DIR / "naver_players_seen.csv"
    naver_players_df = pd.read_csv(players_path, dtype={"naver_pcode": str})
    kaggle_players_df = load_kaggle_player_identities()

    player_map = build_player_id_map(naver_players_df, kaggle_players_df)

    out_path = config.PROCESSED_DIR / "player_id_map.csv"
    player_map.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s rows=%d", out_path, len(player_map))
    logger.info("match_status:\n%s", player_map["match_status"].value_counts(dropna=False))

    unmatched = player_map[player_map["match_status"] == "unmatched"]
    unmatched.to_csv(
        config.PROCESSED_DIR / "player_id_map_unmatched.csv",
        index=False, encoding="utf-8-sig",
    )
    return player_map
