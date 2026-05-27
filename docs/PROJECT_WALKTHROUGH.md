# Project Walkthrough

## 1. Business Framing

The goal is to detect fraudulent credit card transactions in a setting where fraud is extremely rare. This makes ordinary accuracy misleading because a model can classify nearly everything as legitimate and still appear strong.

The useful business question is whether the model can improve fraud capture while keeping false positives at a level a review team can handle.

## 2. Data Source

The project uses the public Kaggle credit card fraud dataset. It contains anonymized PCA-style transaction features `V1` through `V28`, plus `Time`, `Amount`, and the binary target `Class`.

The target values are:

- `0`: legitimate transaction
- `1`: fraudulent transaction

## 3. Exploratory Analysis

The notebook reviews class balance, basic null handling, selected feature relationships, and a correlation heatmap. The key insight is that the minority fraud class is very small, so evaluation must emphasize recall, precision, PR-AUC, and confusion-matrix behavior.

## 4. Modeling Path

The original notebook compares traditional machine learning, ensemble learning, deep learning, and resampling workflows:

- AdaBoost
- Random Forest
- MLP Classifier
- Stacking Classifier
- Voting Classifier
- LSTM
- GRU
- LSTM plus GRU hybrid network
- SMOTEENN-resampled variants

The repo also includes `scripts/train_baseline.py`, which provides a simpler reproducible class-weighted logistic regression baseline.

## 5. Evaluation

Important metrics:

- Precision: how many flagged transactions were actually fraud
- Recall: how much actual fraud was captured
- F1-score: balance between precision and recall
- ROC-AUC: ranking quality across thresholds
- Average precision / PR-AUC: better suited for rare-event detection
- Confusion matrix: operational view of false positives and false negatives

## 6. Deployment Considerations

The Flask app is intentionally lightweight and credential-free. It expects a locally trained model artifact and exposes JSON endpoints for health checks, metadata, and prediction.

For a production fraud system, the next steps would include threshold tuning, monitoring drift, review-queue limits, audit logging, model explainability, and strict handling of payment data.
