#!/usr/bin/env python3
"""What-if 승리확률 변화량 예측 실행 스크립트.

사용 예:
  # 대화형 입력
  uv run python scripts/run_whatif.py

  # JSON 파일로 입력
  uv run python scripts/run_whatif.py --json whatif_input.json

  # 모델 선택 (기본: lgbm)
  uv run python scripts/run_whatif.py --model logistic_model
  uv run python scripts/run_whatif.py --model lgbm_model

  # SHAP 해석 포함
  uv run python scripts/run_whatif.py --json whatif_input.json --shap

JSON 입력 형식:
  {
    "situation": {
      "inning": 7,
      "is_top_bool": 0,
      "outs_before": 1,
      "batting_score_diff_before": -1,
      "runners_on_before": 1,
      "base1_before": 1,
      "base2_before": 0,
      "base3_before": 0,
      "scoring_position_before": 0,
      "late_clutch": 1,
      "is_home_batting": 1
    },
    "actual": {
      "pitcher_pre_era_before": 4.50,
      "pitcher_pre_whip_before": 1.35,
      "pitcher_pre_k9_before": 7.2,
      "pitcher_pre_bb9_before": 3.1,
      "same_hand_matchup": 0,
      "batter_platoon_advantage": 1,
      "batter_pre_avg_before": 0.275,
      "batter_pre_obp_approx_before": 0.345,
      "batter_pre_slg_before": 0.420,
      "batter_pre_ops_before": 0.765
    },
    "candidates": [
      {
        "label": "불펜 A (ERA 3.2)",
        "pitcher_pre_era_before": 3.20,
        "pitcher_pre_whip_before": 1.10,
        "pitcher_pre_k9_before": 9.5,
        "pitcher_pre_bb9_before": 2.4,
        "same_hand_matchup": 1,
        "batter_platoon_advantage": 0,
        "batter_pre_avg_before": 0.275,
        "batter_pre_obp_approx_before": 0.345,
        "batter_pre_slg_before": 0.420,
        "batter_pre_ops_before": 0.765
      }
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_whatif")

# ---------------------------------------------------------------------------
# 대화형 입력 헬퍼
# ---------------------------------------------------------------------------

def _ask(prompt: str, default, cast=float):
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"    ※ 입력 오류, 기본값 {default} 사용")
        return default


def _ask_int(prompt: str, default: int) -> int:
    return int(_ask(prompt, default, cast=lambda x: int(float(x))))


def _ask_float(prompt: str, default: float) -> float:
    return _ask(prompt, default, cast=float)


def interactive_input() -> dict:
    """터미널 대화형 입력으로 situation / actual / candidates 구성."""

    print("\n" + "=" * 55)
    print("  KBO What-if 승리확률 분석기")
    print("=" * 55)

    # ── 경기 상황 ──────────────────────────────────────────
    print("\n[1] 경기 상황")
    inning      = _ask_int("이닝 (1~12)",                    7)
    is_top      = _ask_int("초(1)/말(0)",                    0)
    outs        = _ask_int("현재 아웃 수 (0~2)",              1)
    score_diff  = _ask_int("공격팀 점수차 (공격-수비, 음수=뒤짐)", -1)
    base1       = _ask_int("1루 주자 (0/1)",                  1)
    base2       = _ask_int("2루 주자 (0/1)",                  0)
    base3       = _ask_int("3루 주자 (0/1)",                  0)
    runners_on  = base1 + base2 + base3
    scor_pos    = int(base2 or base3)
    late_clutch = int(inning >= 7 and abs(score_diff) <= 2)
    is_home_bat = int(is_top == 0)

    situation = {
        "inning":                     inning,
        "is_top_bool":                is_top,
        "outs_before":                outs,
        "batting_score_diff_before":  score_diff,
        "runners_on_before":          runners_on,
        "base1_before":               base1,
        "base2_before":               base2,
        "base3_before":               base3,
        "scoring_position_before":    scor_pos,
        "late_clutch":                late_clutch,
        "is_home_batting":            is_home_bat,
    }
    print(f"  → late_clutch={late_clutch}, scoring_position={scor_pos}, runners_on={runners_on}")

    # ── 현재 타자 기록 (공통 사용) ────────────────────────
    print("\n[2] 현재 타자 시즌 기록 (두 투수 비교 시 공통 적용)")
    batter_avg = _ask_float("타율 (avg)",  0.270)
    batter_obp = _ask_float("출루율 (obp)", 0.340)
    batter_slg = _ask_float("장타율 (slg)", 0.400)
    batter_ops = round(batter_obp + batter_slg, 3)
    print(f"  → OPS = {batter_ops}")

    def _pitcher_features(label: str) -> dict:
        print(f"\n[{label}] 투수 시즌 기록")
        era   = _ask_float("ERA",   4.50)
        whip  = _ask_float("WHIP",  1.35)
        k9    = _ask_float("K/9",    7.0)
        bb9   = _ask_float("BB/9",   3.0)
        same  = _ask_int("같은 손잡이 매치업 (0/1)", 0)
        adv   = _ask_int("타자 플래툰 유리 (0/1)",   0)
        return {
            "pitcher_pre_era_before":       era,
            "pitcher_pre_whip_before":      whip,
            "pitcher_pre_k9_before":        k9,
            "pitcher_pre_bb9_before":       bb9,
            "same_hand_matchup":            same,
            "batter_platoon_advantage":     adv,
            "batter_pre_avg_before":        batter_avg,
            "batter_pre_obp_approx_before": batter_obp,
            "batter_pre_slg_before":        batter_slg,
            "batter_pre_ops_before":        batter_ops,
        }

    # ── 현재 투수 ─────────────────────────────────────────
    actual = _pitcher_features("3 — 현재 투수")

    # ── 교체 후보들 ───────────────────────────────────────
    candidates = []
    print("\n[4] 교체 후보 투수 (여러 명 가능)")
    while True:
        n = len(candidates) + 1
        label = input(f"  후보 {n} 이름 (없으면 Enter로 종료): ").strip()
        if not label:
            break
        feats = _pitcher_features(f"후보 {n}: {label}")
        feats["label"] = label
        candidates.append(feats)

    if not candidates:
        print("  후보가 없어 현재 투수 단독 WP만 계산합니다.")

    return {"situation": situation, "actual": actual, "candidates": candidates}


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------

def print_results(actual_wp: float, results: list[dict], shap_df=None) -> None:
    print("\n" + "=" * 55)
    print(f"  현재 투수 승리확률 (WP) : {actual_wp:.4f}  ({actual_wp*100:.1f}%)")
    print("=" * 55)

    if not results:
        return

    print(f"  {'후보':<22} {'WP':>7}  {'ΔWP':>7}  판정")
    print("  " + "-" * 50)
    for r in results:
        direction_kr = {
            "candidate_better": "교체 유리 ▲",
            "actual_better":    "현재 유지 ▼",
            "neutral":          "중립      ─",
        }.get(r["direction"], r["direction"])
        print(f"  {r['label']:<22} {r['wp']:>7.4f}  {r['delta_wp']:>+7.4f}  {direction_kr}")

    best = max(results, key=lambda x: x["wp"])
    print("  " + "-" * 50)
    print(f"  ★ 최선 선택: {best['label']}  (WP {best['wp']:.4f}, ΔWP {best['delta_wp']:+.4f})")

    if shap_df is not None:
        print("\n  [SHAP 피처 기여도 — 현재 투수 기준 상위 8개]")
        print(f"  {'피처':<40} {'SHAP':>8}  {'값':>8}")
        print("  " + "-" * 60)
        for _, row in shap_df.iterrows():
            print(f"  {row['feature']:<40} {row['shap_value']:>8.4f}  {row['feature_value']:>8.3f}")

    print()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KBO What-if 승리확률 분석")
    p.add_argument("--json",  metavar="FILE",
                   help="입력 JSON 파일 경로 (없으면 대화형 입력)")
    p.add_argument("--model", default="lgbm_model",
                   choices=["lgbm_model", "logistic_model"],
                   help="사용할 모델 (기본: lgbm_model)")
    p.add_argument("--feature-mode", default="mvp",
                   choices=["mvp", "advanced"],
                   help="피처 세트 (기본: mvp)")
    p.add_argument("--shap", action="store_true",
                   help="현재 투수 SHAP 기여도 출력 (lgbm 전용)")
    p.add_argument("--save", metavar="FILE",
                   help="결과를 JSON 파일로 저장")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 엔진 로드 ──────────────────────────────────────────
    logger.info("모델 로드: %s (feature_mode=%s)", args.model, args.feature_mode)
    try:
        from src.models.whatif_engine import WhatIfEngine
        engine = WhatIfEngine.load(model_name=args.model, feature_mode=args.feature_mode)
    except FileNotFoundError:
        logger.error(
            "모델 파일을 찾을 수 없습니다: models/%s.pkl\n"
            "먼저 run_train_models.py 를 실행하세요.", args.model
        )
        sys.exit(1)
    logger.info("모델 로드 완료")

    # ── 입력 ───────────────────────────────────────────────
    if args.json:
        path = Path(args.json)
        if not path.exists():
            logger.error("JSON 파일 없음: %s", path)
            sys.exit(1)
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("JSON 입력 로드: %s", path)
    else:
        data = interactive_input()

    situation  = data["situation"]
    actual     = data["actual"]
    candidates = data.get("candidates", [])

    # label 키를 피처 dict에서 분리
    cand_labels = [c.pop("label", f"후보{i+1}") for i, c in enumerate(candidates)]

    # ── 예측 ───────────────────────────────────────────────
    actual_wp = engine.predict_single(situation, actual)

    results = []
    for label, cand_feats in zip(cand_labels, candidates):
        r = engine.predict_delta(situation, actual, cand_feats, candidate_label=label)
        results.append({
            "label":     label,
            "wp":        r["candidate_wp"],
            "delta_wp":  r["delta_wp"],
            "direction": r["direction"],
        })
    results.sort(key=lambda x: x["delta_wp"], reverse=True)

    # ── SHAP ───────────────────────────────────────────────
    shap_df = None
    if args.shap:
        if args.model != "lgbm_model":
            logger.warning("SHAP은 lgbm_model 전용입니다. --model lgbm_model 로 실행하세요.")
        else:
            try:
                shap_df = engine.get_shap_explanation(situation, actual, n_top=8)
            except ImportError:
                logger.warning("shap 패키지 필요: uv add shap")

    # ── 출력 ───────────────────────────────────────────────
    print_results(actual_wp, results, shap_df)

    # ── 저장 ───────────────────────────────────────────────
    if args.save:
        output = {
            "actual_wp": actual_wp,
            "results":   results,
        }
        Path(args.save).write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("결과 저장: %s", args.save)


if __name__ == "__main__":
    main()
