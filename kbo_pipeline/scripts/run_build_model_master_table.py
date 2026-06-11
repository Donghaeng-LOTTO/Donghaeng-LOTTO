"""모델 학습용 PA 단위 마스터 테이블 생성 스크립트.

사용 예:
  uv run python scripts/run_build_model_master_table.py
  uv run python scripts/run_build_model_master_table.py --output data/processed/model_master_pa.csv
  uv run python scripts/run_build_model_master_table.py --no-eligible-file
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.model_master_table import run_model_master_table  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_build_model_master_table")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=None,
        help="전체 마스터 테이블 저장 경로. 기본값: data/processed/model_master_pa.csv",
    )
    parser.add_argument(
        "--eligible-output",
        default=None,
        help="학습 권장 행 저장 경로. 기본값: data/processed/model_master_pa_eligible.csv",
    )
    parser.add_argument(
        "--no-eligible-file",
        action="store_true",
        help="model_master_pa_eligible.csv 저장을 생략한다.",
    )
    args = parser.parse_args()

    run_model_master_table(
        output_path=args.output,
        eligible_output_path=args.eligible_output,
        write_eligible=not args.no_eligible_file,
    )
    logger.info("done.")


if __name__ == "__main__":
    main()
