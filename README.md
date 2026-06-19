# 🏦 Fraud Detection

Binary classification model for detecting fraudulent credit card transactions using XGBoost, Logistic Regression, and MLP.

## 📊 Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds (0.17%).

Features V1-V28 are PCA-transformed for confidentiality. `Time` is seconds from first transaction. `Amount` is transaction amount.

## 🛠 Pipeline

1. **Data Cleaning** — Removed 1,081 duplicates
2. **Feature Engineering** — `Amount` → `Amount_log`, `Time` → `Hours`
3. **SMOTE** — Balanced classes to 50/50 (453,204 samples)
4. **Modeling** — Logistic Regression, XGBoost, MLP
5. **Evaluation** — ROC-AUC, Recall, Precision, Confusion Matrix

## 🏆 Results

| Model | ROC-AUC | Recall | Precision | FP |
|-------|:------:|:-----:|:--------:|:--:|
| Logistic Regression | 0.959 | 0.85 | 0.12 | 586 |
| MLP | 0.965 | 0.87 | 0.06 | 1419 |
| **XGBoost** | **0.976** | **0.80** | **0.85** | **13** |

**XGBoost** was chosen as the final model:
- Catches **80%** of fraud with only **13 false alarms** out of 56,746 transactions
- Precision 85% means minimal customer disruption
- Trains in 3 seconds vs 4 minutes for Random Forest

Voting ensemble was tested but did not improve over XGBoost alone.

### Train and save model

```python
import joblib
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
joblib.dump(model, 'models/xgboost_fraud.pkl')
