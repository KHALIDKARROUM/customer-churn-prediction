from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_code_hash() -> str:
    digest = hashlib.sha256()
    for path in (BASE_DIR / "model_training.py", BASE_DIR / "train_model.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


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
        training_code_hash=training_code_hash(),
        git_revision=git_revision(),
        training_run_id=uuid4().hex,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
    joblib.dump(artifact, temporary_model_path, compress=3)
    os.replace(temporary_model_path, MODEL_PATH)

    metadata = {
        "artifact_version": artifact["artifact_version"],
        "trained_at_utc": artifact["trained_at_utc"],
        "sklearn_version": artifact["sklearn_version"],
        "runtime_environment": artifact["runtime_environment"],
        "training_code_hash": artifact["training_code_hash"],
        "git_revision": artifact["git_revision"],
        "training_run_id": artifact["training_run_id"],
        "artifact_sha256": file_hash(MODEL_PATH),
        "data_hash": artifact["data_hash"],
        "model_name": artifact["model_name"],
        "model_selection": artifact["model_selection"],
        "decision_threshold": artifact["decision_threshold"],
        "holdout_size": artifact["holdout_size"],
        "metrics": artifact["metrics"],
        "model_comparison": artifact["model_comparison"].to_dict("records"),
    }
    write_json_atomically(METADATA_PATH, metadata)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Selected model: {artifact['model_name']}")
    print(f"Decision threshold: {artifact['decision_threshold']:.3f}")
    for name, value in artifact["metrics"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
