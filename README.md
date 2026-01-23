# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques under extreme class imbalance conditions.

## Problem
Fraudulent transactions account for a very small fraction of total transactions, making accuracy a misleading metric. The objective is to maximize fraud detection (recall) while controlling false positives.

## Dataset
- Source: Kaggle Credit Card Fraud Detection dataset
- Note: Dataset is not included in this repository due to GitHub file size limits.  
  Download it from Kaggle and place it in the `data/` directory.

## Approach
- Data cleaning and preprocessing
- Handling class imbalance
- Model training and evaluation
- Performance analysis using Precision, Recall, F1-score, and PR-AUC

## Models
- Logistic Regression
- Random Forest (or your actual model)

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

## Results
- Improved recall for fraud detection
- Demonstrated trade-off between fraud detection and false positives

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
