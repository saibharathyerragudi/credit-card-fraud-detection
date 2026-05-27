from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.schema import FEATURE_COLUMNS, TARGET_COLUMN


def build_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=None,
                    random_state=42,
                ),
            ),
        ]
    )


def load_dataset(path: Path):
    import pandas as pd

    data = pd.read_csv(path)
    expected_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(expected_columns.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    clean = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    return clean[FEATURE_COLUMNS], clean[TARGET_COLUMN].astype(int)


def evaluate_model(model, x_test, y_test) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["not_fraud", "fraud"],
            zero_division=0,
            output_dict=True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline credit card fraud model.")
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/fraud_detection_pipeline.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/metrics/baseline_metrics.json"))
    args = parser.parse_args()

    import joblib
    from sklearn.model_selection import train_test_split

    x, y = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to {args.model_out}")
    print(f"Metrics saved to {args.metrics_out}")
    print(json.dumps({key: metrics[key] for key in ["precision", "recall", "f1", "roc_auc", "average_precision"]}, indent=2))


if __name__ == "__main__":
    main()
