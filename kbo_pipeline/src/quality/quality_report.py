"""데이터셋 품질 리포트.

경기 단위 체크:
- record raw 존재 / games_detail 행 존재
- relay raw 존재 / events 개수 / PA 개수
- 재구성 점수 vs record 최종 점수 일치 여부
- state warn 비율
산출물: data/processed/dataset_quality_report.csv
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def build_quality_report() -> pd.DataFrame:
    games = pd.read_csv(config.PROCESSED_DIR / "games.csv", dtype={"game_id": str})

    detail_path = config.PROCESSED_DIR / "games_detail.csv"
    detail = pd.read_csv(detail_path, dtype={"game_id": str}) if detail_path.exists() else pd.DataFrame()

    events_path = config.PROCESSED_DIR / "relay_events.csv"
    pa_path = config.PROCESSED_DIR / "plate_appearances.csv"

    events_counts = pd.Series(dtype=int)
    if events_path.exists():
        ev = pd.read_csv(events_path, dtype={"game_id": str},
                         usecols=["game_id", "seqno"])
        events_counts = ev.groupby("game_id")["seqno"].count()

    pa_counts = pd.Series(dtype=int)
    warn_ratio = pd.Series(dtype=float)
    score_valid = pd.DataFrame()
    if pa_path.exists():
        pa = pd.read_csv(pa_path, dtype={"game_id": str})
        pa_counts = pa.groupby("game_id")["relay_no"].count()
        if "state_parse_status" in pa.columns:
            warn_ratio = (
                pa.assign(is_warn=pa["state_parse_status"].ne("ok"))
                .groupby("game_id")["is_warn"].mean()
            )

    rows = []
    for _, g in games.iterrows():
        game_id = str(g["game_id"])

        record_raw = (config.RAW_RECORD_DIR / game_id / f"{game_id}_record.json").exists()
        relay_dir = config.RAW_RELAY_DIR / game_id
        relay_innings = len(list(relay_dir.glob(f"{game_id}_inning_*.json"))) if relay_dir.exists() else 0

        d = detail[detail["game_id"] == game_id] if not detail.empty else pd.DataFrame()
        has_detail = len(d) > 0
        away_score = d["away_score"].iloc[0] if has_detail else None
        home_score = d["home_score"].iloc[0] if has_detail else None
        status_code = d["status_code"].iloc[0] if has_detail else g.get("status_code")
        cancel_flag = d["cancel_flag"].iloc[0] if has_detail else g.get("cancel_flag")

        n_events = int(events_counts.get(game_id, 0))
        n_pa = int(pa_counts.get(game_id, 0))
        w_ratio = float(warn_ratio.get(game_id, float("nan"))) if len(warn_ratio) else None

        # 휴리스틱: 정규 9이닝 경기 PA는 보통 55~110개
        pa_count_plausible = (50 <= n_pa <= 130) if n_pa else False

        rows.append({
            "game_id": game_id,
            "game_date": g.get("game_date"),
            "status_code": status_code,
            "cancel_flag": cancel_flag,

            "has_record_raw": record_raw,
            "has_games_detail": has_detail,
            "relay_inning_files": relay_innings,
            "has_relay_raw": relay_innings > 0,

            "n_relay_events": n_events,
            "n_plate_appearances": n_pa,
            "pa_count_plausible": pa_count_plausible,
            "state_warn_ratio": w_ratio,

            "away_score_record": away_score,
            "home_score_record": home_score,
        })

    report = pd.DataFrame(rows)

    # 점수 재구성 검증 결과 merge (run_build_features에서 생성)
    valid_path = config.PROCESSED_DIR / "score_validation.csv"
    if valid_path.exists():
        sv = pd.read_csv(valid_path, dtype={"game_id": str})
        report = report.merge(sv, on="game_id", how="left")

    out_path = config.PROCESSED_DIR / "dataset_quality_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[saved] %s rows=%d", out_path, len(report))

    # 요약 출력
    if not report.empty:
        logger.info("record raw coverage: %.1f%%", 100 * report["has_record_raw"].mean())
        logger.info("relay raw coverage:  %.1f%%", 100 * report["has_relay_raw"].mean())
        logger.info("PA plausible ratio:  %.1f%%", 100 * report["pa_count_plausible"].mean())

    return report
