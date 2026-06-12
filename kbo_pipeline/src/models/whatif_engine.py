"""What-if 승리확률 변화량 예측 엔진.

핵심 개념 ([문2016] WV = 타격 후 WE - 타격 전 WE 확장):
  - 실제 선택의 예측 WP  : 실제 선수/작전 피처로 모델 추론
  - 가상 선택의 예측 WP  : 교체 후보 피처로 모델 재추론
  - ΔWP = 가상 WP - 실제 WP  (양수 = 가상 선택이 더 유리)

[Hirotsu2005] DP 구조에서 "교체 후보별 WP 비교 최대화" 를 ML 추론으로 구현.
[Bukiet1997] 선수 교체 전후 기대승수 차이 계산의 ML 버전.

SHAP 기반 해석:
  - 교체 전후 SHAP 값 차이로 "어떤 피처가 WP 변화를 이끌었는가" 설명 가능.
  - 팬 대상 서비스: "투수 교체 시 +X% 이유: ERA 차이 때문"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.feature_config import (
    ADVANCED_FEATURES,
    MVP_FEATURES,
    filter_available_features,
)
from src.models.train import load_model, prepare_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 엔진 클래스
# ---------------------------------------------------------------------------
class WhatIfEngine:
    """What-if 승리확률 변화량 예측 엔진.

    사용 예시:
        engine = WhatIfEngine.load()

        # 현재 타석 상황
        situation = {
            'inning': 7, 'is_top_bool': 0, 'outs_before': 1,
            'batting_score_diff_before': -1,
            'runners_on_before': 1, 'base1_before': 1,
            'base2_before': 0, 'base3_before': 0,
            'state_we': 0.42, 'state_re': 0.55,
            ...
        }
        # 실제 선수 피처
        actual_player   = {'batter_pre_ops_before': 0.750, ...}
        # 대체 선수 피처
        candidate_player = {'batter_pre_ops_before': 0.820, ...}

        result = engine.predict_delta(situation, actual_player, candidate_player)
        print(result)
        # {'actual_wp': 0.42, 'candidate_wp': 0.48, 'delta_wp': +0.06}
    """

    def __init__(self, model: Any, feature_list: list[str], model_type: str = "lgbm"):
        self.model        = model
        self.feature_list = feature_list
        self.model_type   = model_type
        self._shap_explainer = None

    @classmethod
    def load(
        cls,
        model_name: str = "lgbm_model",
        feature_mode: str = "mvp",
    ) -> "WhatIfEngine":
        """저장된 모델 파일로 엔진 초기화."""
        model = load_model(model_name)
        feature_list = MVP_FEATURES if feature_mode == "mvp" else ADVANCED_FEATURES
        model_type = "lgbm" if "lgbm" in model_name else "logistic"
        return cls(model, feature_list, model_type)

    # ------------------------------------------------------------------
    # 단일 타석 예측
    # ------------------------------------------------------------------
    def _row_to_df(self, situation: dict, player_features: dict) -> pd.DataFrame:
        """상황 + 선수 피처를 하나의 DataFrame 행으로 합친다."""
        combined = {**situation, **player_features}
        df = pd.DataFrame([combined])
        available = filter_available_features(list(df.columns), self.feature_list, verbose=False)
        # 없는 피처는 NaN으로 채움
        for f in self.feature_list:
            if f not in df.columns:
                df[f] = np.nan
        return df[available]

    def predict_single(self, situation: dict, player_features: dict) -> float:
        """단일 상황에서 승리확률 예측 (0~1)."""
        X = self._row_to_df(situation, player_features)
        for col in X.select_dtypes(include="bool").columns:
            X[col] = X[col].astype(int)
        for col in X.select_dtypes(include="boolean").columns:
            X[col] = X[col].fillna(-1).astype(int)

        if self.model_type == "lgbm":
            prob = float(self.model.predict(X)[0])
        else:
            prob = float(self.model.predict_proba(X)[0, 1])

        return round(np.clip(prob, 0.0, 1.0), 4)

    # ------------------------------------------------------------------
    # What-if 비교
    # ------------------------------------------------------------------
    def predict_delta(
        self,
        situation: dict,
        actual_features: dict,
        candidate_features: dict,
        candidate_label: str = "후보",
    ) -> dict:
        """실제 선택 vs 가상 선택의 승리확률 비교.

        Args:
            situation:           경기 상황 피처 dict (이닝, 아웃, 점수차 등).
            actual_features:     실제 선수/작전 피처 dict.
            candidate_features:  가상 선수/작전 피처 dict.
            candidate_label:     가상 선택 이름 (UI 표시용).

        Returns:
            {
                'actual_wp':    0.42,
                'candidate_wp': 0.48,
                'delta_wp':    +0.06,
                'candidate_label': '후보 이름',
                'direction':   'candidate_better',  # or 'actual_better' or 'neutral'
            }
        """
        actual_wp    = self.predict_single(situation, actual_features)
        candidate_wp = self.predict_single(situation, candidate_features)
        delta        = round(candidate_wp - actual_wp, 4)

        direction = (
            "candidate_better" if delta > 0.005
            else "actual_better" if delta < -0.005
            else "neutral"
        )

        return {
            "actual_wp":       actual_wp,
            "candidate_wp":    candidate_wp,
            "delta_wp":        delta,
            "candidate_label": candidate_label,
            "direction":       direction,
        }

    def predict_best_candidate(
        self,
        situation: dict,
        actual_features: dict,
        candidates: list[dict],
        labels: list[str] | None = None,
    ) -> dict:
        """여러 후보 중 최선 교체 선택을 찾는다.

        Args:
            situation:    경기 상황 피처 dict.
            actual_features: 현재 선수 피처 dict.
            candidates:   후보 선수 피처 dict 목록.
            labels:       후보 이름 목록.

        Returns:
            {
                'actual_wp': float,
                'results': [{'label', 'wp', 'delta_wp', 'direction'}, ...],
                'best': {'label', 'wp', 'delta_wp'},
            }
        """
        if labels is None:
            labels = [f"후보{i+1}" for i in range(len(candidates))]

        actual_wp = self.predict_single(situation, actual_features)

        results = []
        for label, cand_feats in zip(labels, candidates):
            cand_wp  = self.predict_single(situation, cand_feats)
            delta    = round(cand_wp - actual_wp, 4)
            direction = (
                "candidate_better" if delta > 0.005
                else "actual_better" if delta < -0.005
                else "neutral"
            )
            results.append({
                "label":     label,
                "wp":        cand_wp,
                "delta_wp":  delta,
                "direction": direction,
            })

        results_sorted = sorted(results, key=lambda x: x["delta_wp"], reverse=True)
        best = results_sorted[0] if results_sorted else {}

        return {
            "actual_wp": actual_wp,
            "results":   results_sorted,
            "best":      best,
        }

    # ------------------------------------------------------------------
    # SHAP 해석
    # ------------------------------------------------------------------
    def get_shap_explanation(
        self,
        situation: dict,
        player_features: dict,
        n_top: int = 10,
    ) -> pd.DataFrame:
        """단일 예측에 대한 SHAP 기반 피처 기여도 반환.

        Returns:
            피처명 / shap_value / feature_value 컬럼 DataFrame (상위 n_top).
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap 설치 필요: pip install shap")

        if self._shap_explainer is None:
            if self.model_type == "lgbm":
                self._shap_explainer = shap.TreeExplainer(self.model)
            else:
                raise NotImplementedError("Logistic 모델의 SHAP은 미구현")

        X = self._row_to_df(situation, player_features)
        for col in X.select_dtypes(include="bool").columns:
            X[col] = X[col].astype(int)

        shap_values = self._shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # binary: index 1 = positive class
        else:
            sv = shap_values[0]

        df_shap = pd.DataFrame({
            "feature":       X.columns,
            "shap_value":    sv,
            "feature_value": X.iloc[0].values,
        }).sort_values("shap_value", key=abs, ascending=False)

        return df_shap.head(n_top).reset_index(drop=True)

    def explain_delta(
        self,
        situation: dict,
        actual_features: dict,
        candidate_features: dict,
        n_top: int = 8,
    ) -> pd.DataFrame:
        """교체 전후 SHAP 차이로 WP 변화 원인 설명.

        Returns:
            피처명 / shap_actual / shap_candidate / shap_delta 컬럼 DataFrame.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap 설치 필요: pip install shap")

        if self._shap_explainer is None and self.model_type == "lgbm":
            import shap
            self._shap_explainer = shap.TreeExplainer(self.model)

        def _get_shap(player_feats: dict) -> np.ndarray:
            X = self._row_to_df(situation, player_feats)
            for col in X.select_dtypes(include="bool").columns:
                X[col] = X[col].astype(int)
            sv = self._shap_explainer.shap_values(X)
            if isinstance(sv, list):
                return sv[1][0]
            return sv[0]

        shap_actual    = _get_shap(actual_features)
        shap_candidate = _get_shap(candidate_features)

        X_actual = self._row_to_df(situation, actual_features)

        df = pd.DataFrame({
            "feature":         X_actual.columns,
            "shap_actual":     shap_actual,
            "shap_candidate":  shap_candidate,
            "shap_delta":      shap_candidate - shap_actual,
        }).sort_values("shap_delta", key=abs, ascending=False)

        return df.head(n_top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 배치 What-if (경기 로그 전체에 적용)
# ---------------------------------------------------------------------------
def run_batch_whatif(
    master_df: pd.DataFrame,
    engine: WhatIfEngine,
    actual_col_prefix: str = "",
    candidate_col_prefix: str = "candidate_",
    situation_cols: list[str] | None = None,
) -> pd.DataFrame:
    """마스터 테이블 전체에 대해 실제 vs 가상 WP를 일괄 계산.

    서비스 DB에 미리 계산해두거나 분석 리포트 생성에 사용.

    Args:
        master_df: 확장 마스터 테이블 (상황 + 실제 + 가상 피처 모두 포함).
        engine:    WhatIfEngine 인스턴스.
        situation_cols: 상황 피처 컬럼 목록 (None → STATE_FEATURES 기본값).
        actual_col_prefix:    실제 선수 피처 컬럼 접두사.
        candidate_col_prefix: 가상 선수 피처 컬럼 접두사.

    Returns:
        actual_wp / candidate_wp / delta_wp 컬럼이 추가된 DataFrame.
    """
    from src.models.feature_config import STATE_FEATURES

    if situation_cols is None:
        situation_cols = STATE_FEATURES

    results = []
    for _, row in master_df.iterrows():
        sit  = {k: row.get(k) for k in situation_cols}
        act  = {k.replace(actual_col_prefix, ""): row.get(k)
                for k in master_df.columns if k.startswith(actual_col_prefix) and k in engine.feature_list}
        cand = {k.replace(candidate_col_prefix, ""): row.get(k)
                for k in master_df.columns if k.startswith(candidate_col_prefix)}

        actual_wp    = engine.predict_single(sit, act)
        candidate_wp = engine.predict_single(sit, cand) if cand else np.nan
        delta_wp     = round(candidate_wp - actual_wp, 4) if not np.isnan(candidate_wp) else np.nan

        results.append({"actual_wp": actual_wp, "candidate_wp": candidate_wp, "delta_wp": delta_wp})

    out = master_df.copy()
    out[["actual_wp", "candidate_wp", "delta_wp"]] = pd.DataFrame(results, index=master_df.index)
    return out
