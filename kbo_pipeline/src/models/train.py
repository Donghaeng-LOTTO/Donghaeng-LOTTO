"""모델 학습 파이프라인.

전처리 정책 (모델별 다름):
  LightGBM:
    - 스케일링 불필요 (트리는 크다/작다 비교 → 절대값 스케일 무관)
    - 결측치 불필요 (NaN을 자체 분기로 처리)
    - 범주형 처리: lgb.Dataset categorical_feature 파라미터
    - float32 변환으로 메모리 절반 절약 권장
    - free_raw_data=True로 Dataset 생성 후 원본 해제

  LogisticRegression (베이스라인):
    - StandardScaler 필수 (경사 기반 최적화는 스케일 민감)
    - SimpleImputer 필수
    - solver='saga' 필수 (lbfgs는 100만+ 행에 매우 느림)
    - SGDClassifier(loss='log_loss')로 대체하면 30~90초로 단축 가능

학습 전략:
  - 시간 기반 분할 (Time-based split): 마지막 2시즌 = 검증
  - 과거 → 미래 예측 구조에서 k-fold는 시간 누수 발생

출력:
  - models/lgbm_model.pkl
  - models/logistic_model.pkl
  - models/feature_importance.csv
  - models/evaluation_report.json
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.feature_config import (
    ADVANCED_FEATURES,
    CATEGORICAL_FEATURES,
    MVP_FEATURES,
    filter_available_features,
)

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def load_training_data(
    use_extended: bool = True,
    label_col: str = "batting_team_win_label",
    usecols_extra: list[str] | None = None,
) -> pd.DataFrame:
    """학습용 CSV 로드.

    필요한 컬럼만 읽어 IO 시간 단축.
    extended(신규 피처 포함) 우선, 없으면 base eligible로 fallback.
    """
    ext_path  = config.PROCESSED_DIR / "model_master_pa_extended_eligible.csv"
    base_path = config.PROCESSED_DIR / "model_master_pa_eligible.csv"
    path = ext_path if (use_extended and ext_path.exists()) else base_path

    if not path.exists():
        raise FileNotFoundError(f"학습 데이터 없음: {path}")

    logger.info("데이터 로드: %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("로드 완료: %d행 x %d열", len(df), len(df.columns))

    df = df[df[label_col].notna()].copy()
    logger.info("라벨 유효 행: %d", len(df))
    return df


def prepare_features(
    df: pd.DataFrame,
    feature_list: list[str],
    label_col: str = "batting_team_win_label",
    to_float32: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """피처 행렬 X, 라벨 y 반환.

    Args:
        to_float32: True면 float32 변환 → 메모리 절반 절약.
                    LightGBM / LogisticRegression 모두 float32 지원.
    """
    available = filter_available_features(list(df.columns), feature_list)
    logger.info("사용 피처: %d / %d개", len(available), len(feature_list))

    X = df[available].copy()

    # bool/boolean → int8 변환
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype("int8")
    for col in X.select_dtypes(include="boolean").columns:
        X[col] = X[col].fillna(-1).astype("int8")

    # float64 → float32: 메모리 절반, 정확도 손실 없음
    if to_float32:
        for col in X.select_dtypes(include="float64").columns:
            X[col] = X[col].astype("float32")

    y = df[label_col].astype("float32")
    return X, y


def time_based_split(
    df: pd.DataFrame,
    test_seasons: list[int] | None = None,
    n_test_seasons: int = 2,
) -> tuple[pd.Index, pd.Index]:
    """시즌 기반 학습/검증 분할 (시간 누수 방지)."""
    if "season" not in df.columns:
        df = df.copy()
        df["season"] = pd.to_numeric(
            df["game_id"].astype(str).str[:4], errors="coerce"
        )

    seasons_sorted = sorted(df["season"].dropna().unique())
    if len(seasons_sorted) < 2:
        logger.warning("시즌 1개 — 80/20 랜덤 분할로 대체")
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
        return train_idx, test_idx

    if test_seasons is None:
        test_seasons = seasons_sorted[-n_test_seasons:]

    train_idx = df[~df["season"].isin(test_seasons)].index
    test_idx  = df[ df["season"].isin(test_seasons)].index

    logger.info(
        "학습 시즌: %s (%d행) | 검증 시즌: %s (%d행)",
        [s for s in seasons_sorted if s not in test_seasons],
        len(train_idx), test_seasons, len(test_idx),
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# 전처리 파이프라인
# ---------------------------------------------------------------------------
def build_logistic_pipeline(n_samples: int):
    """로지스틱 회귀 전처리 파이프라인.

    스케일링이 왜 필요한가:
      경사 기반 최적화는 피처 스케일에 민감하다.
      ERA(0~9), inning(1~15), cum_ab(0~700)가 섞이면
      그래디언트 스텝이 발산하거나 수렴이 매우 느려진다.
      StandardScaler로 평균 0, 분산 1로 맞추면 해결.

    solver 선택:
      - lbfgs: 기본값, 소규모 데이터에 적합, 100만+ 행에 매우 느림
      - saga : 확률적 경사하강 기반, 대규모 데이터 최적, L1/L2 모두 지원
      - n_jobs=-1로 멀티코어 병렬화
    """
    try:
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
    except ImportError:
        raise ImportError("pip install scikit-learn")

    # 170만행 초과면 SGDClassifier(log_loss)로 대체 (수십 초 수준)
    if n_samples > 500_000:
        logger.info(
            "샘플 수 %d > 500,000 → SGDClassifier(log_loss) 사용 (saga보다 빠름)",
            n_samples,
        )
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=100,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    else:
        clf = LogisticRegression(
            solver="saga",      # ← lbfgs 대신 saga: 대규모 데이터에 적합
            max_iter=300,
            C=1.0,
            n_jobs=-1,
            random_state=42,
        )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # NaN → 중앙값
        ("scaler",  StandardScaler()),                  # 피처 스케일 통일
        ("clf",     clf),
    ])


# ---------------------------------------------------------------------------
# 로지스틱 회귀 학습
# ---------------------------------------------------------------------------
def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """로지스틱 회귀 (SGDClassifier) 학습 및 평가.

    [김이2016] 근거: KBO 실증에서 로지스틱 모형이 피타고라스 정리보다 우수.
    베이스라인으로 LightGBM 대비 성능 기준선 제공.
    """
    pipe = build_logistic_pipeline(n_samples=len(X_train))
    pipe.fit(X_train, y_train)

    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        # SGDClassifier는 decision_function → sigmoid 변환
        from scipy.special import expit
        proba = expit(pipe.decision_function(X_test))

    metrics = _compute_metrics(y_test, proba, model_name="Logistic/SGD")
    logger.info("Logistic 평가: AUC=%.4f  Brier=%.4f", metrics.get("auc", 0), metrics.get("brier", 0))
    return {"model": pipe, "metrics": metrics, "proba_test": proba}


# ---------------------------------------------------------------------------
# LightGBM 학습
# ---------------------------------------------------------------------------
def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: list[str] | None = None,
    params: dict | None = None,
) -> dict:
    """LightGBM 학습 및 평가.

    스케일링이 왜 불필요한가:
      트리는 피처값의 대소(크다/작다)만 비교해 분기점을 찾는다.
      ERA=4.5를 표준화해서 -0.2로 바꿔도 분기 순서가 동일하므로
      결과가 완전히 같다. 스케일링은 시간 낭비.

    결측치가 왜 불필요한가:
      LightGBM은 NaN을 별도 분기로 자동 처리한다.
      imputer를 추가하면 오히려 정보를 훼손할 수 있다.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    default_params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 50,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "lambda_l2":         0.1,
        "verbose":           -1,
        "random_state":      42,
        "n_jobs":            -1,
    }
    if params:
        default_params.update(params)

    cat_feats = [f for f in (categorical_features or CATEGORICAL_FEATURES)
                 if f in X_train.columns]

    # 범주형 컬럼 int 변환 (NaN은 -1로 채워 LightGBM이 인식하도록)
    for col in cat_feats:
        X_train[col] = X_train[col].fillna(-1).astype("int8")
        X_test[col]  = X_test[col].fillna(-1).astype("int8")

    # free_raw_data=True: Dataset 내부 bin 변환 완료 후 원본 DataFrame 참조 해제
    # → 학습 중 약 270MB 추가 메모리 절약
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_feats or "auto",
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        X_test, label=y_test,
        reference=dtrain,
        categorical_feature=cat_feats or "auto",
        free_raw_data=True,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    proba = model.predict(X_test, num_iteration=model.best_iteration)
    metrics = _compute_metrics(y_test, proba, model_name="LightGBM")
    logger.info("LightGBM 평가: AUC=%.4f  Brier=%.4f", metrics.get("auc", 0), metrics.get("brier", 0))

    fi = pd.DataFrame({
        "feature":            model.feature_name(),
        "importance_gain":    model.feature_importance(importance_type="gain"),
        "importance_split":   model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    return {"model": model, "metrics": metrics, "proba_test": proba, "feature_importance": fi}


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def _compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    except ImportError:
        return {"model": model_name, "error": "scikit-learn 없음"}

    mask = y_true.isin([0.0, 1.0])
    y_b  = y_true[mask]
    p_b  = y_pred[mask.values] if hasattr(mask, 'values') else y_pred[mask]

    metrics: dict = {
        "model":       model_name,
        "n_samples":   len(y_true),
        "n_binary":    int(mask.sum()),
        "mean_pred":   round(float(np.mean(y_pred)), 4),
        "mean_actual": round(float(y_true.mean()), 4),
    }

    if mask.sum() > 0:
        metrics["auc"]      = round(float(roc_auc_score(y_b, p_b)), 4)
        metrics["brier"]    = round(float(brier_score_loss(y_b, p_b)), 4)
        metrics["log_loss"] = round(float(log_loss(y_b, p_b)), 4)

    # 캘리브레이션 (10분위)
    try:
        df_cal = pd.DataFrame({"pred": y_pred[mask.values], "actual": y_b.values})
        df_cal["bucket"] = pd.qcut(df_cal["pred"], q=10, labels=False, duplicates="drop")
        metrics["calibration_by_decile"] = (
            df_cal.groupby("bucket")
            .agg(mean_pred=("pred","mean"), mean_actual=("actual","mean"), n=("pred","count"))
            .round(4).to_dict(orient="records")
        )
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# 저장 / 로드
# ---------------------------------------------------------------------------
def save_model(obj, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("[saved] %s", path)
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 풀 학습 파이프라인
# ---------------------------------------------------------------------------
def run_training(
    feature_mode: str = "mvp",
    label_col: str = "batting_team_win_label",
    test_seasons: list[int] | None = None,
    lgbm_params: dict | None = None,
) -> dict:
    """전체 학습 파이프라인.

    Args:
        feature_mode : 'mvp' (~40피처) 또는 'advanced' (~70피처).
        label_col    : 학습 타깃.
        test_seasons : 검증 시즌. None → 마지막 2시즌 자동 선택.
        lgbm_params  : LightGBM 파라미터 override.
    """
    feature_list = MVP_FEATURES if feature_mode == "mvp" else ADVANCED_FEATURES
    use_extended = feature_mode == "advanced"

    df = load_training_data(use_extended=use_extended, label_col=label_col)
    X, y = prepare_features(df, feature_list, label_col=label_col, to_float32=True)

    train_idx, test_idx = time_based_split(df, test_seasons=test_seasons)
    X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
    y_train, y_test = y.loc[train_idx],         y.loc[test_idx]

    logger.info(
        "학습셋: %d행  검증셋: %d행  피처: %d개  메모리(float32): %.0fMB",
        len(X_train), len(X_test), len(X.columns),
        X_train.memory_usage(deep=True).sum() / 1e6,
    )

    results: dict = {"feature_list": list(X.columns)}

    # ── 1) 로지스틱 베이스라인 ──
    logger.info("=== 로지스틱 회귀 / SGD 베이스라인 ===")
    try:
        lr_result = train_logistic(X_train, y_train, X_test, y_test)
        save_model(lr_result["model"], "logistic_model")
        results["logistic"] = lr_result
    except Exception as e:
        logger.error("로지스틱 실패: %s", e, exc_info=True)

    # ── 2) LightGBM 주 모델 ──
    logger.info("=== LightGBM 주 모델 ===")
    try:
        lgbm_result = train_lgbm(
            X_train.copy(), y_train,
            X_test.copy(),  y_test,
            params=lgbm_params,
        )
        save_model(lgbm_result["model"], "lgbm_model")
        fi_path = MODELS_DIR / "feature_importance.csv"
        lgbm_result["feature_importance"].to_csv(fi_path, index=False)
        logger.info("[saved] %s", fi_path)
        results["lgbm"] = lgbm_result
    except Exception as e:
        logger.error("LightGBM 실패: %s", e, exc_info=True)

    # ── 평가 리포트 저장 ──
    report = {
        "logistic_metrics": results.get("logistic", {}).get("metrics"),
        "lgbm_metrics":     results.get("lgbm", {}).get("metrics"),
        "feature_mode":     feature_mode,
        "label_col":        label_col,
        "n_features":       len(results["feature_list"]),
        "n_train":          len(train_idx),
        "n_test":           len(test_idx),
        "note": {
            "lgbm_preprocessing":    "없음 — 트리는 스케일 무관, NaN 자동 처리",
            "logistic_preprocessing": "StandardScaler + SimpleImputer + saga solver",
            "memory_optimization":   "float32 변환으로 float64 대비 50% 절약",
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("[saved] %s", report_path)

    return results
