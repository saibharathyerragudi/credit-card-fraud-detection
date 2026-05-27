# Credit Card Fraud Detection

An end-to-end fraud analytics portfolio project using the public European credit card transactions dataset to analyze class imbalance, compare fraud detection models, and package a safer inference workflow for review.

This repository is structured as a portfolio case study: the modeling notebook is included, a reproducible baseline training script is provided, and the documentation explains the business problem, model choices, evaluation metrics, and deployment considerations.

## Project At A Glance

| Area | Details |
|---|---|
| Domain | Financial fraud detection |
| Dataset | Kaggle credit card fraud transactions |
| Objective | Detect rare fraudulent transactions while managing false positives |
| Primary Challenge | Severe class imbalance |
| Deliverables | Modeling notebook, baseline training pipeline, Flask inference API, research paper, presentation deck |
| Key Techniques | Feature scaling, class-weighted classification, resampling experiments, ensemble models, precision/recall analysis |

## Business Objective

Credit card fraud teams need a model that can identify suspicious transactions early without overwhelming analysts with unnecessary reviews. Because fraud is rare, raw accuracy is not enough; a model can look accurate while missing the transactions that matter most.

This project answers:

- How imbalanced is the fraud detection problem?
- Which modeling approaches improve fraud recall?
- What trade-offs exist between recall, precision, false positives, and operational review workload?
- How can a trained model be packaged safely for a lightweight API workflow?

## Dataset

Source: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Expected local file path:

```text
data/creditcard.csv
```

The raw CSV is intentionally excluded from Git because of size. Download it from Kaggle and place it at the path above before running the notebook or training script.

## Notebook

Open the main notebook here:

```text
notebooks/credit_card_fraud_detection_modeling.ipynb
```

The notebook explores:

- Class distribution and correlation analysis
- Mutual-information feature selection
- Baseline machine learning models
- Ensemble classifiers
- Deep learning experiments with LSTM and GRU layers
- SMOTEENN resampling comparisons
- Accuracy, precision, recall, F1, sensitivity, specificity, and ROC-AUC

## Reproducible Baseline

Run a clean baseline pipeline without editing the notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_baseline.py --data data/creditcard.csv
```

This trains a class-weighted logistic regression pipeline and writes:

```text
models/fraud_detection_pipeline.joblib
outputs/metrics/baseline_metrics.json
```

The baseline is intentionally simple and interpretable. Use the notebook for the broader ensemble/deep learning experiments.

## Flask Inference API

After training the baseline model:

```bash
python app.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Prediction example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"Time":0,"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0,"Amount":0}}'
```

## Repository Structure

```text
credit-card-fraud-detection/
├── README.md
├── app.py
├── requirements.txt
├── requirements-optional.txt
├── data/
│   ├── .gitkeep
│   └── README.md
├── docs/
│   ├── APP_SECURITY_NOTES.md
│   ├── MODELING_APPROACH.md
│   ├── PROJECT_WALKTHROUGH.md
│   ├── fraud_detection_presentation.pptx
│   └── fraud_detection_research_paper.docx
├── models/
│   └── .gitkeep
├── notebooks/
│   └── credit_card_fraud_detection_modeling.ipynb
├── outputs/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
├── scripts/
│   └── train_baseline.py
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       └── schema.py
└── tests/
    └── test_schema.py
```

## Skills Demonstrated

- Fraud analytics and imbalanced classification framing
- Precision, recall, F1, sensitivity, specificity, ROC-AUC, and PR-AUC evaluation
- Resampling workflow with SMOTEENN
- Ensemble modeling and deep learning experimentation
- Reproducible training script design
- Safe API packaging without hardcoded credentials or local user databases
