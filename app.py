from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.fraud_detection.schema import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = Path("models/fraud_detection_pipeline.joblib")

app = Flask(__name__)


def get_model_path() -> Path:
    return Path(os.getenv("FRAUD_MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model():
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python scripts/train_baseline.py --data data/creditcard.csv` first."
        )
    return joblib.load(model_path)


def parse_features(payload: dict) -> pd.DataFrame:
    if "features" not in payload:
        raise ValueError("Request JSON must include a `features` object.")

    features = payload["features"]
    if not isinstance(features, dict):
        raise ValueError("`features` must be a JSON object keyed by feature name.")

    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    ordered = {column: float(features[column]) for column in FEATURE_COLUMNS}
    return pd.DataFrame([ordered], columns=FEATURE_COLUMNS)


@app.get("/")
def index():
    return jsonify(
        {
            "project": "Credit Card Fraud Detection",
            "endpoints": ["/health", "/metadata", "/predict"],
            "model_path": str(get_model_path()),
        }
    )


@app.get("/health")
def health():
    model_path = get_model_path()
    return jsonify({"status": "ok", "model_available": model_path.exists()})


@app.get("/metadata")
def metadata():
    return jsonify(
        {
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "target": "Class",
            "positive_class": "Fraudulent transaction",
        }
    )


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        features = parse_features(payload)
        model = load_model()
        prediction = int(model.predict(features)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])

        return jsonify(
            {
                "prediction": prediction,
                "label": "fraud" if prediction == 1 else "not_fraud",
                "fraud_probability": probability,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
