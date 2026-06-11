#!/usr/bin/env python3
"""모델 학습 스크립트.

전제 조건:
  - run_build_all_features.py 완료 (model_master_pa_extended_eligible.csv)
  - pip install lightgbm scikit-learn shap

사용법:
  cd kbo_pipeline
  python scripts/run_train_models.py

  # 고도화 피처로 학습:
  python scripts/run_train_models.py --feature-mode advanced

  # 검증 시즌 직접 지정:
  python scripts/run_train_models.py --test-seasons 2024 2025

  # 롯데 전용 라벨로 학습:
  python scripts/run_train_models.py --label lotte_win_label

  # What-if 엔진 테스트 실행:
  python scripts/run_train_models.py --test-whatif
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_train_models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KBO What-if 모델 학습")
    p.add_argument(
        "--feature-mode",
        choices=["mvp", "advanced"],
        default="mvp",
        help="피처 세트 선택 (기본: mvp)",
    )
    p.add_argument(
        "--label",
        default="batting_team_win_label",
        help="타깃 라벨 컬럼 (기본: batting_team_win_label)",
    )
    p.add_argument(
        "--test-seasons",
        nargs="+",
        type=int,
        default=None,
        help="검증 시즌 (미지정 시 마지막 2시즌)",
    )
    p.add_argument(
        "--test-whatif",
        action="store_true",
        help="학습 후 What-if 엔진 동작 테스트 실행",
    )
    p.add_argument(
        "--lgbm-num-leaves",   type=int,   default=63,   help="LightGBM num_leaves")
    p.add_argument(
        "--lgbm-lr",           type=float, default=0.05, help="LightGBM learning_rate")
    p.add_argument(
        "--lgbm-min-child",    type=int,   default=50,   help="LightGBM min_child_samples")
    return p.parse_args()


def test_whatif_engine() -> None:
    """학습된 모델로 What-if 엔진 동작 확인."""
    from src.models.whatif_engine import WhatIfEngine

    logger.info("=== What-if 엔진 동작 테스트 ===")
    engine = WhatIfEngine.load(model_name="lgbm_model", feature_mode="mvp")

    # 가상 시나리오: 7회말 1점 뒤짐, 1사 1루
    situation = {
        "inning":                     7,
        "is_top_bool":                0,   # 홈팀 공격(말)
        "outs_before":                1,
        "batting_score_diff_before": -1,
        "runners_on_before":          1,
        "base1_before":               1,
        "base2_before":               0,
        "base3_before":               0,
        "scoring_position_before":    0,
        "late_clutch":                1,
        "is_home_batting":            1,
        "state_we":                   0.42,  # WE 룩업 테이블 조인 전 임시값
        "state_re":                   0.55,
    }

    # 현재 투수 (ERA 4.5, WHIP 1.35)
    actual_pitcher = {
        "pitcher_pre_era_before":  4.50,
        "pitcher_pre_whip_before": 1.35,
        "pitcher_pre_k9_before":   7.2,
        "pitcher_pre_bb9_before":  3.1,
        "same_hand_matchup":       0,
        "batter_platoon_advantage": 1,
        "batter_pre_avg_before":   0.275,
        "batter_pre_obp_approx_before": 0.345,
        "batter_pre_slg_before":   0.420,
        "batter_pre_ops_before":   0.765,
    }

    # 교체 후보 투수 (ERA 3.2, WHIP 1.10)
    candidate_pitcher = {
        "pitcher_pre_era_before":  3.20,
        "pitcher_pre_whip_before": 1.10,
        "pitcher_pre_k9_before":   9.5,
        "pitcher_pre_bb9_before":  2.4,
        "same_hand_matchup":       1,
        "batter_platoon_advantage": 0,
        "batter_pre_avg_before":   0.275,
        "batter_pre_obp_approx_before": 0.345,
        "batter_pre_slg_before":   0.420,
        "batter_pre_ops_before":   0.765,
    }

    result = engine.predict_delta(
        situation,
        actual_pitcher,
        candidate_pitcher,
        candidate_label="ERA 3.2 불펜투수",
    )

    logger.info("What-if 테스트 결과:")
    logger.info("  현재 투수 WP     : %.4f", result["actual_wp"])
    logger.info("  교체 후보 WP     : %.4f", result["candidate_wp"])
    logger.info("  ΔWP (가상-실제)  : %+.4f", result["delta_wp"])
    logger.info("  판정             : %s", result["direction"])

    # 여러 후보 비교
    candidates = [
        {**candidate_pitcher, "pitcher_pre_era_before": 2.80, "pitcher_pre_whip_before": 1.05},
        {**candidate_pitcher, "pitcher_pre_era_before": 3.80, "pitcher_pre_whip_before": 1.20},
        {**candidate_pitcher, "pitcher_pre_era_before": 5.10, "pitcher_pre_whip_before": 1.50},
    ]
    best_result = engine.predict_best_candidate(
        situation,
        actual_pitcher,
        candidates,
        labels=["후보A(ERA 2.8)", "후보B(ERA 3.8)", "후보C(ERA 5.1)"],
    )
    logger.info("\n다중 후보 비교:")
    logger.info("  현재 WP: %.4f", best_result["actual_wp"])
    for r in best_result["results"]:
        logger.info("  %-20s WP=%.4f  ΔWP=%+.4f  [%s]",
                    r["label"], r["wp"], r["delta_wp"], r["direction"])
    logger.info("  최선 선택: %s", best_result["best"].get("label"))


def main() -> None:
    args = parse_args()

    lgbm_params = {
        "num_leaves":        args.lgbm_num_leaves,
        "learning_rate":     args.lgbm_lr,
        "min_child_samples": args.lgbm_min_child,
    }

    logger.info("=" * 60)
    logger.info("모델 학습 시작")
    logger.info("  피처 모드  : %s", args.feature_mode)
    logger.info("  타깃 라벨  : %s", args.label)
    logger.info("  검증 시즌  : %s", args.test_seasons or "마지막 2시즌 자동 선택")
    logger.info("=" * 60)

    from src.models.train import run_training

    results = run_training(
        feature_mode=args.feature_mode,
        label_col=args.label,
        test_seasons=args.test_seasons,
        lgbm_params=lgbm_params,
    )

    # 결과 요약 출력
    logger.info("=" * 60)
    logger.info("학습 완료 요약")
    if "logistic" in results:
        m = results["logistic"]["metrics"]
        logger.info("  Logistic   AUC=%.4f  Brier=%.4f  LogLoss=%.4f",
                    m.get("auc", 0), m.get("brier", 0), m.get("log_loss", 0))
    if "lgbm" in results:
        m = results["lgbm"]["metrics"]
        logger.info("  LightGBM   AUC=%.4f  Brier=%.4f  LogLoss=%.4f",
                    m.get("auc", 0), m.get("brier", 0), m.get("log_loss", 0))
        fi = results["lgbm"]["feature_importance"]
        logger.info("  Top 10 피처 (gain):")
        for _, row in fi.head(10).iterrows():
            logger.info("    %-45s gain=%d", row["feature"], int(row["importance_gain"]))
    logger.info("  사용 피처 수: %d", len(results.get("feature_list", [])))
    logger.info("=" * 60)

    # What-if 엔진 테스트
    if args.test_whatif:
        test_whatif_engine()


if __name__ == "__main__":
    main()
