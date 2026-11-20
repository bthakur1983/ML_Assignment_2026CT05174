# BITS M.Tech Assignment 2: Income Classification

**Student ID:** 2026CT05174  
**GitHub repository:** [ML_Assignment_2026CT05174](https://github.com/bthakur1983/ML_Assignment_2026CT05174)

## Problem statement

This project predicts whether an adult's annual income exceeds $50,000. It is a binary classification task; income above $50,000 is the positive class (`1`).

## Dataset description

The Adult Census Income dataset is obtained from OpenML and originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). It contains 48,842 instances and 14 input features, with a mix of numerical and categorical data. The target column, `class`, is converted to a binary label. The dataset includes missing-value markers and an imbalanced target.

The notebook uses scikit-learn's `fetch_openml`. Internet access is needed for the initial download; scikit-learn caches the dataset for later use.

## Preprocessing and evaluation

- Missing-value markers are normalized and missing values are imputed.
- Duplicate rows are identified and removed.
- The data is split into 80% training and 20% testing with stratification and random seed `42`.
- Numeric features use median imputation and standard scaling.
- Categorical features use most-frequent imputation and one-hot encoding.
- Preprocessing is part of each scikit-learn pipeline, so it is fitted on training data only.
- Each model is evaluated with Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC).

## Models and configurations

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"`, `random_state=42` |
| Decision Tree | `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42` |
| k-Nearest Neighbors | `n_neighbors=7`, `weights="distance"` |
| Gaussian Naive Bayes | `GaussianNB` with default parameters |
| Random Forest (Ensemble) | `n_estimators=200`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1` |

Other estimator parameters use scikit-learn defaults.

## Results

The table reports held-out test-set performance. The overall winner is selected by MCC because it accounts for all four confusion-matrix outcomes and is informative for an imbalanced binary target.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.9022 | 0.5602 | 0.8527 | 0.6762 | 0.5692 |
| Decision Tree | 0.7982 | 0.8942 | 0.5500 | 0.8643 | 0.6722 | 0.5653 |
| kNN | 0.8305 | 0.8669 | 0.6607 | 0.6002 | 0.6290 | 0.5205 |
| Naive Bayes | 0.6173 | 0.8329 | 0.3777 | 0.9242 | 0.5363 | 0.3855 |
| Random Forest (Ensemble) | 0.8193 | 0.9148 | 0.5847 | 0.8467 | 0.6917 | 0.5895 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong baseline, with AUC 0.9022 and MCC 0.5692. Its recall (0.8527) is high relative to its precision (0.5602). |
| Decision Tree | Achieves recall 0.8643 and MCC 0.5653, close to Logistic Regression, while retaining a slightly lower accuracy of 0.7982. |
| kNN | Has the highest accuracy (0.8305) and precision (0.6607), but the lowest recall among these models (0.6002), so it misses more positive cases. |
| Naive Bayes | Has the highest recall (0.9242), but low precision (0.3777), accuracy (0.6173), and MCC (0.3855), indicating many false-positive predictions. |
| Random Forest (Ensemble) | Achieves the best AUC (0.9148), F1 (0.6917), and MCC (0.5895), with a strong recall of 0.8467. |

**Overall winner:** Random Forest (Ensemble), based on the highest MCC (0.5895).

## Graph representations

The notebook presents classification reports and confusion matrices for each model, a graph comparing all six evaluation metrics, and ROC curves.

![Comparison graph of all six metrics](model_metrics_comparison.png)

![ROC curves for the five classifiers](roc_curves.png)

## Repository contents

- `requirements.txt`: Python dependencies.
- `README.md`: assignment summary and results.
- `test_data.csv`: held-out test examples and labels.
- `model/2026CT05174_ML_Assignment.ipynb`: source notebook implementing all five classifiers.
- `model_comparison.csv`: full-precision metric results.
- `model_metrics_comparison.png` and `roc_curves.png`: generated figures shown above.

## Execution instructions

Use Python 3.12 or newer. From the repository root, create and activate a virtual environment, then install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

In Jupyter, open `model/2026CT05174_ML_Assignment.ipynb` and run the cells from top to bottom. Start Jupyter from the repository root so generated model and result artifacts are saved in the expected project folders. Internet access is needed for the initial OpenML download.
