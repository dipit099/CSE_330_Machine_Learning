# PRACTICE 3 — Exam Simulation (bare blanks, no hints)
# Style: exactly like the real exam skeleton from C-Online 3
# Fill every blank, then run to verify
#
# RULES: don't look at practice_1 until you've tried everything yourself!
# =============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import xgboost as xgb
import random
from scipy import stats

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# TODO: load the dataset
data = ___


# TODO: separate features (X) and target
# The species columns are one-hot encoded: species_0, species_1, species_2
# Convert them back to a single target column

X = ___

y_onehot = ___
target_names = ___
cols_to_label = {'species_0': 0, 'species_1': 1, 'species_2': 2}
target = ___


# TODO: train-test split (80%-20%)
X_train, X_test, y_train, y_test = ___


# TODO: scale the features
scalar = ___
X_train_normalized = ___
X_test_normalized = ___


# -------------------- Single Logistic Regression --------------------

# TODO: train a single Logistic Regression (max_iter=1000, random_state=42) and compute lr_acc

simple_reg = ___
___
simple_reg_preds = ___
lr_acc = ___


# -------------------- Bagging with Logistic Regression --------------------

n_estimators = 10
models = []

# TODO: train n_estimators models on bootstrap samples
n_samples = len(X_train_normalized)

for i in range(n_estimators):
    X_sample, y_sample = ___
    boot_model = ___
    ___
    models.append(boot_model)


# TODO: collect predictions from all models
predictions = []

for model in models:
    preds = ___
    predictions.append(preds)


# TODO: majority voting and compute bagging_acc
predictions = np.array(predictions)
vote_preds = ___
bagging_acc = ___


# -------------------- XGBoost --------------------

n_estimators_list = [5, 20, 50]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

xgb_poor_acc = 1.0
xgb_best_acc = 0.0
worst_config = {}
best_config = {}

# TODO: train and predict with each combination
for n_estimator in n_estimators_list:
    for max_depth in max_depth_list:
        for learning_rate in learning_rate_list:

            xgb_model = ___
            ___
            xgb_preds = ___
            xgb_acc = ___

            if xgb_acc < xgb_poor_acc:
                xgb_poor_acc = xgb_acc
                worst_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

            if xgb_acc > xgb_best_acc:
                xgb_best_acc = xgb_acc
                best_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

print(f"Worst Config: {worst_config}")
print(f"Best Config:  {best_config}")

# -------------------- Final Output --------------------
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Single Logistic Regression Accuracy: {round(lr_acc, 4)}")
print(f"Bagging (Logistic Regression) Accuracy: {round(bagging_acc, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(xgb_poor_acc, 4)}")
print(f"XGBoost Best Accuracy: {round(xgb_best_acc, 4)}")
print("=" * 60)
