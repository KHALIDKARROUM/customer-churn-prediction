from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_training import train_model_artifact


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
FEATURE_FIELDS = [
    "tenure",
    "Contract",
    "TotalCharges",
    "InternetService",
    "MonthlyCharges",
    "PaymentMethod",
]
NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"]


def data_file_hash() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].replace(" ", np.nan),
        errors="coerce",
    )
    return data.dropna().reset_index(drop=True)


def main() -> None:
    data_hash = data_file_hash()
    artifact = train_model_artifact(
        load_training_data(),
        FEATURE_FIELDS,
        NUMERIC_FIELDS,
        random_state=42,
        data_hash=data_hash,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)

    metadata = {
        "artifact_version": artifact["artifact_version"],
        "trained_at_utc": artifact["trained_at_utc"],
        "sklearn_version": artifact["sklearn_version"],
        "data_hash": artifact["data_hash"],
        "model_name": artifact["model_name"],
        "model_selection": artifact["model_selection"],
        "decision_threshold": artifact["decision_threshold"],
        "holdout_size": artifact["holdout_size"],
        "metrics": artifact["metrics"],
        "model_comparison": artifact["model_comparison"].to_dict("records"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Selected model: {artifact['model_name']}")
    print(f"Decision threshold: {artifact['decision_threshold']:.3f}")
    for name, value in artifact["metrics"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
