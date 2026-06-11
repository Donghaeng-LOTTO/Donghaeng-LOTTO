#!/usr/bin/env python3
"""신규 피처 일괄 빌드 스크립트.

실행 순서:
  1. WE / RE 룩업 테이블
  2. Recency 이동평균 + 좌우 스플릿
  3. 투수 컨텍스트 + 파크팩터
  4. 확장 마스터 테이블 (model_master_pa_extended.csv)

전제 조건 (먼저 실행 필요):
  - run_collect_2008_2025.py   (raw 데이터 수집)
  - run_build_features.py      (plate_appearances, boxscores 등 기본 CSV)
  - run_build_model_master_table.py (model_master_pa.csv)

사용법:
  cd kbo_pipeline
  python scripts/run_build_all_features.py

  # WE 룩업 테이블을 특정 학습 시즌으로만 산출 (누수 방지):
  python scripts/run_build_all_features.py --train-seasons 2008 2009 ... 2023
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# kbo_pipeline 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_build_all_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="신규 피처 일괄 빌드")
    p.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=None,
        help="WE/RE 룩업 계산에 사용할 학습 시즌 (미지정 시 전체). 예: 2008 2009 ... 2023",
    )
    p.add_argument(
        "--skip-we",      action="store_true", help="WE/RE 테이블 생성 건너뜀")
    p.add_argument(
        "--skip-recency", action="store_true", help="Recency/Split 통계 생성 건너뜀")
    p.add_argument(
        "--skip-context", action="store_true", help="투수 컨텍스트/파크팩터 생성 건너뜀")
    p.add_argument(
        "--skip-master",  action="store_true", help="확장 마스터 테이블 생성 건너뜀")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. WE / RE 룩업 테이블
    if not args.skip_we:
        logger.info("=" * 60)
        logger.info("STEP 1: WE / RE 룩업 테이블 생성")
        logger.info("=" * 60)
        from src.features.we_re_table import run_we_re_tables
        run_we_re_tables(train_seasons=args.train_seasons)
    else:
        logger.info("STEP 1 건너뜀 (--skip-we)")

    # 2. Recency 이동평균 + 좌우 스플릿
    if not args.skip_recency:
        logger.info("=" * 60)
        logger.info("STEP 2: Recency 이동평균 + 좌우 스플릿 생성")
        logger.info("=" * 60)
        from src.features.recency_split_stats import run_recency_and_split_stats
        run_recency_and_split_stats(windows=[5, 10])
    else:
        logger.info("STEP 2 건너뜀 (--skip-recency)")

    # 3. 투수 컨텍스트 + 파크팩터
    if not args.skip_context:
        logger.info("=" * 60)
        logger.info("STEP 3: 투수 컨텍스트 + 파크팩터 생성")
        logger.info("=" * 60)
        from src.features.pitcher_context_park import run_pitcher_context_and_park_factors
        run_pitcher_context_and_park_factors()
    else:
        logger.info("STEP 3 건너뜀 (--skip-context)")

    # 4. 확장 마스터 테이블
    if not args.skip_master:
        logger.info("=" * 60)
        logger.info("STEP 4: 확장 마스터 테이블 생성")
        logger.info("=" * 60)
        from src.features.model_master_table_ext import run_extended_master_table
        extended = run_extended_master_table()
        logger.info("확장 마스터 완성: %d행 x %d열", len(extended), len(extended.columns))
    else:
        logger.info("STEP 4 건너뜀 (--skip-master)")

    logger.info("=" * 60)
    logger.info("모든 피처 빌드 완료")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
