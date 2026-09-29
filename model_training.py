from __future__ import annotations

from datetime import datetime, timezone
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


ARTIFACT_VERSION = 3


def runtime_environment() -> dict[str, str]:
    """Return the versions that must match when the artifact is loaded."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def build_pipeline(
    estimator: Any,
    numeric_fields: list[str],
    categorical_fields: list[str],
) -> Pipeline:
    """Build one fitted preprocessing/model unit for training and inference."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", MinMaxScaler(), numeric_fields),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_fields,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _candidate_pipelines(
    numeric_fields: list[str],
    categorical_fields: list[str],
    random_state: int,
) -> dict[str, Pipeline]:
    estimators = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
    return {
        name: build_pipeline(estimator, numeric_fields, categorical_fields)
        for name, estimator in estimators.items()
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        np.finfo(float).eps,
    )
    return float(np.clip(thresholds[int(np.nanargmax(f1_values))], 0.1, 0.9))


def _classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Brier score": brier_score_loss(y_true, probabilities),
    }
    return metrics, confusion_matrix(y_true, predictions)


def _original_feature_name(
    encoded_name: str,
    numeric_fields: list[str],
    categorical_fields: list[str],
) -> str:
    clean_name = encoded_name.split("__", maxsplit=1)[-1]
    if clean_name in numeric_fields:
        return clean_name
    for original in categorical_fields:
        if clean_name.startswith(f"{original}_"):
            return original
    return clean_name


def _feature_importance_tables(
    fitted_pipeline: Pipeline,
    numeric_fields: list[str],
    categorical_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    estimator = fitted_pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        values = np.zeros(len(feature_names), dtype=float)

    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": values}
    ).sort_values("Importance", ascending=False)
    feature_importance["OriginalFeature"] = feature_importance["Feature"].map(
        lambda name: _original_feature_name(
            name,
            numeric_fields,
            categorical_fields,
        )
    )
    grouped_importance = (
        feature_importance.groupby("OriginalFeature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )
    return feature_importance, grouped_importance


def train_model_artifact(
    data: pd.DataFrame,
    feature_fields: list[str],
    numeric_fields: list[str],
    random_state: int = 42,
    data_hash: str | None = None,
    training_code_hash: str | None = None,
    git_revision: str | None = None,
    training_run_id: str | None = None,
) -> dict[str, Any]:
    """Select on training CV, evaluate once on holdout, and fit production model."""
    categorical_fields = [field for field in feature_fields if field not in numeric_fields]
    x = data[feature_fields].copy()
    y = (data["Churn"] == "Yes").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    candidates = _candidate_pipelines(
        numeric_fields,
        categorical_fields,
        random_state,
    )
    selection_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "Accuracy": make_scorer(accuracy_score),
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1-score": make_scorer(f1_score, zero_division=0),
    }
    comparison_rows: list[dict[str, Any]] = []
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=selection_cv,
            scoring=scoring,
            n_jobs=1,
        )
        row: dict[str, Any] = {"Model": name}
        for metric in scoring:
            row[metric] = float(np.mean(scores[f"test_{metric}"]))
        row["F1 std"] = float(np.std(scores["test_F1-score"]))
        comparison_rows.append(row)

    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        "F1-score", ascending=False
    )
    model_name = str(model_comparison.iloc[0]["Model"])
    selected_pipeline = candidates[model_name]

    calibration_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    calibrated_template = CalibratedClassifierCV(
        estimator=selected_pipeline,
        method="sigmoid",
        cv=calibration_cv,
        n_jobs=1,
    )
    training_probabilities = cross_val_predict(
        calibrated_template,
        x_train,
        y_train,
        cv=calibration_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    decision_threshold = _best_f1_threshold(y_train, training_probabilities)

    evaluation_model = clone(calibrated_template).fit(x_train, y_train)
    test_probabilities = evaluation_model.predict_proba(x_test)[:, 1]
    metrics, matrix = _classification_metrics(
        y_test,
        test_probabilities,
        decision_threshold,
    )

    production_model = clone(calibrated_template).fit(x, y)
    oof_probabilities = cross_val_predict(
        calibrated_template,
        x,
        y,
        cv=calibration_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    explanation_pipeline = clone(selected_pipeline).fit(x, y)
    feature_importance, grouped_importance = _feature_importance_tables(
        explanation_pipeline,
        numeric_fields,
        categorical_fields,
    )

    return {
        "artifact_version": ARTIFACT_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "runtime_environment": runtime_environment(),
        "training_code_hash": training_code_hash,
        "git_revision": git_revision,
        "training_run_id": training_run_id,
        "data_hash": data_hash,
        "model_name": model_name,
        "model_selection": "Highest mean 5-fold training-set CV F1",
        "pipeline": production_model,
        "decision_threshold": decision_threshold,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "model_comparison": model_comparison,
        "feature_importance": feature_importance,
        "grouped_importance": grouped_importance,
        "oof_probabilities": oof_probabilities,
        "feature_fields": feature_fields,
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "holdout_size": len(y_test),
    }
