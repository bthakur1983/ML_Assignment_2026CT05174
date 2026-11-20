# BITS M.Tech Assignment 2: Income Classification

**Student ID:** 2026CT05174  
**GitHub repository:** https://github.com/bthakur1983/ML_Assignment_2026CT05174

## Problem statement

Build and evaluate machine-learning classifiers that predict whether an adult's income exceeds $50,000 per year. The task is binary classification; the positive class is represented by `income_gt_50k = 1`.

## Dataset

This project uses the Adult Census Income dataset from OpenML (originally from the UCI Machine Learning Repository). It contains 48,842 instances and 14 input features, with both numeric and categorical data. The target column `class` is converted to a binary label. The dataset includes missing-value markers and an imbalanced target.

The notebook loads the dataset through scikit-learn's `fetch_openml`. Internet access is needed on first download; scikit-learn caches the dataset for later runs.

## Method

- Train/test split: 80%/20%, stratified by target, with random seed 42.
- Numeric features: median imputation followed by standard scaling.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Preprocessing is fitted within each model pipeline using training data only.
- Classifiers: Logistic Regression, Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes, and Random Forest.
- Metrics: Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC).

## Results

Scores below are from the held-out test split. The overall winner is selected by MCC, which considers all four confusion-matrix outcomes and is useful for an imbalanced binary target.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (Ensemble) | 0.8193 | 0.9148 | 0.5847 | 0.8467 | 0.6917 | 0.5895 |
| Logistic Regression | 0.8045 | 0.9022 | 0.5602 | 0.8527 | 0.6762 | 0.5692 |
| Decision Tree | 0.7982 | 0.8942 | 0.5500 | 0.8643 | 0.6722 | 0.5653 |
| kNN | 0.8305 | 0.8669 | 0.6607 | 0.6002 | 0.6290 | 0.5205 |
| Naive Bayes | 0.6173 | 0.8329 | 0.3777 | 0.9242 | 0.5363 | 0.3855 |

### Observations

- Random Forest has the best AUC (0.9148), F1 (0.6917), and MCC (0.5895), making it the overall winner under the stated MCC selection rule.
- kNN has the highest accuracy (0.8305) and precision (0.6607), but its recall is lower (0.6002).
- Naive Bayes has the highest recall (0.9242), but substantially lower precision and MCC; many positive predictions are false positives.
- Logistic Regression and Decision Tree provide competitive recall and MCC, while Random Forest has the strongest overall balance under MCC.

## Visualizations

The executed notebook includes per-model classification reports and confusion matrices, a comparison plot of all six metrics, and ROC curves.

![Metric comparison](model_metrics_comparison.png)

![ROC curves](roc_curves.png)

## Repository contents

- `2026CT05174_ML_Assignment.ipynb`: source notebook.
- `2026CT05174_ML_Assignment.executed.ipynb`: notebook with execution outputs.
- `test_data.csv`: held-out test examples and labels.
- `model_comparison.csv`: full-precision metric results.
- `model/`: five serialized fitted pipelines (`.joblib`).
- `model_metrics_comparison.png`, `roc_curves.png`: generated figures.

## Setup and execution

Use Python 3.12 or newer, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook 2026CT05174_ML_Assignment.ipynb
```

Run notebook cells from top to bottom. On first run, allow internet access for the OpenML dataset download. The notebook creates the `model/` directory and saves model and result artifacts in the project folder.
