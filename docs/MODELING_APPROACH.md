# Modeling Approach

## Why Accuracy Is Not Enough

Credit card fraud detection is a rare-event classification problem. If fraud represents less than 1% of transactions, a naive model can predict every transaction as legitimate and still score above 99% accuracy.

That is why this project prioritizes:

- Recall for fraud capture
- Precision for analyst workload control
- F1-score for balance
- ROC-AUC and PR-AUC for ranking quality
- Confusion matrix counts for operational interpretation

## Notebook Experiments

The notebook explores two broad tracks.

### Without Resampling

Models are trained on the natural class distribution. This reflects the real population but can bias models toward the majority class.

### With SMOTEENN

SMOTEENN combines synthetic minority oversampling with edited nearest-neighbor cleaning. This can increase fraud recall, but the resulting model must be reviewed carefully because synthetic sampling can change the data distribution seen during training.

## Baseline Script

`scripts/train_baseline.py` is intentionally simpler than the notebook. It trains a class-weighted logistic regression model using all Kaggle transaction features.

This gives the repository a reproducible baseline that is:

- Fast to run
- Easy to inspect
- Easy to compare against the notebook's more complex models
- Suitable for powering the lightweight Flask API

## Recommended Improvements

Future versions could add:

- Threshold tuning for review capacity
- PR curve and ROC curve exports
- SHAP or coefficient-based feature interpretation
- Time-based validation split
- Model registry metadata
- Batch scoring script for new transactions
