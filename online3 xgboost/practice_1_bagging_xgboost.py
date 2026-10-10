# PRACTICE 1 — Bagging with Logistic Regression + XGBoost
# Style: step-by-step, simple, easy to remember
# Dataset: modified_large_iris_dataset.csv (one-hot encoded target)
#
# HOW TO USE:
#   - Read each TODO and the BLANK line below it
#   - Try to fill in the blank yourself first
#   - Then check the SOLUTION comment right below
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

# =============================================================
# STEP 1: Load the dataset
# =============================================================
# TODO: load the CSV file into a dataframe called 'data'

data = pd.read_csv("C-Online 3/modified_large_iris_dataset.csv")
# SOLUTION: data = pd.read_csv("filename.csv")

print(data.shape)
print(data.head())

# =============================================================
# STEP 2: Separate features (X) and target
# =============================================================
# The target is split across 3 columns: species_0, species_1, species_2
# (True/False one-hot encoding)
# We need ONE column with values 0, 1, or 2

# TODO: drop the species columns from data to get X
X = data.drop(columns=['species_0', 'species_1', 'species_2'])
# SOLUTION: X = data.drop(columns=['species_0', 'species_1', 'species_2'])

# TODO: get just the species columns from data
y_onehot = data[['species_0', 'species_1', 'species_2']]
# SOLUTION: y_onehot = data[['species_0', 'species_1', 'species_2']]

# TODO: convert one-hot → label using idxmax, then map to 0/1/2
target_names = y_onehot.idxmax(axis=1)
# SOLUTION: target_names = y_onehot.idxmax(axis=1)
# idxmax gives: 'species_0', 'species_1', or 'species_2' for each row

cols_to_label = {'species_0': 0, 'species_1': 1, 'species_2': 2}
target = target_names.map(cols_to_label)
# SOLUTION: target = target_names.map({'species_0': 0, 'species_1': 1, 'species_2': 2})

print(target.value_counts())

# =============================================================
# STEP 3: Train-test split (80% train, 20% test)
# =============================================================
# TODO: split X and target into train/test sets

X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8, random_state=42)
# SOLUTION: X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8, random_state=42)

print("Train size:", X_train.shape, "  Test size:", X_test.shape)

# =============================================================
# STEP 4: Scale the features
# =============================================================
# TODO: create scaler, fit on train, transform both

scalar = StandardScaler()
# SOLUTION: scalar = StandardScaler()

X_train_normalized = scalar.fit_transform(X_train)
# SOLUTION: X_train_normalized = scalar.fit_transform(X_train)
# fit_transform = learn mean/std FROM train, then scale

X_test_normalized = scalar.transform(X_test)
# SOLUTION: X_test_normalized = scalar.transform(X_test)
# transform only = use the SAME mean/std learned from train (NO re-fitting!)

# =============================================================
# STEP 5: Single Logistic Regression
# =============================================================
# TODO: create model, train, predict, compute accuracy

simple_reg = LogisticRegression(max_iter=1000, random_state=42)
# SOLUTION: simple_reg = LogisticRegression(max_iter=1000, random_state=42)

simple_reg.fit(X_train_normalized, y_train)
# SOLUTION: simple_reg.fit(X_train_normalized, y_train)

simple_reg_preds = simple_reg.predict(X_test_normalized)
# SOLUTION: simple_reg_preds = simple_reg.predict(X_test_normalized)

lr_acc = accuracy_score(y_test, simple_reg_preds)
# SOLUTION: lr_acc = accuracy_score(y_test, simple_reg_preds)

print(f"Single LR Accuracy: {lr_acc:.4f}")

# =============================================================
# STEP 6: Bagging with Logistic Regression
# =============================================================
# Bagging = train same model on DIFFERENT random bootstrap samples,
#           then combine by majority vote

n_estimators = 10
models = []

# PART A: Train models on bootstrap samples
# TODO: loop n_estimators times, resample, train, append

n_samples = len(X_train_normalized)

for i in range(n_estimators):
    # resample = randomly pick n_samples rows WITH replacement
    X_sample, y_sample = resample(X_train_normalized, y_train, replace=True, n_samples=n_samples, random_state=i)
    # SOLUTION: X_sample, y_sample = resample(X_train_normalized, y_train, replace=True, n_samples=n_samples, random_state=i)

    boot_model = LogisticRegression(max_iter=1000, random_state=42)
    boot_model.fit(X_sample, y_sample)
    models.append(boot_model)

# PART B: Collect predictions from each model
# TODO: loop through models and get predictions

predictions = []

for model in models:
    preds = model.predict(X_test_normalized)
    # SOLUTION: preds = model.predict(X_test_normalized)
    predictions.append(preds)

# PART C: Majority voting
# TODO: convert list → array, then use stats.mode to get majority vote

predictions = np.array(predictions)
# SOLUTION: predictions = np.array(predictions)
# Shape is [n_estimators, n_test_samples] = [10, 30000]
# Each row = one model's predictions for all test samples

vote_preds = stats.mode(predictions, axis=0).mode
# SOLUTION: vote_preds = stats.mode(predictions, axis=0).mode
# axis=0 = look DOWN the columns → for each test sample, find the most common prediction

bagging_acc = accuracy_score(y_test, vote_preds)
# SOLUTION: bagging_acc = accuracy_score(y_test, vote_preds)

print(f"Bagging LR Accuracy: {bagging_acc:.4f}")

# =============================================================
# STEP 7: XGBoost — try all hyperparameter combinations
# =============================================================
# TODO: fill in the XGBClassifier inside the loop

n_estimators_list = [5, 20, 50]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

xgb_poor_acc = 1.0
xgb_best_acc = 0.0
worst_config = {}
best_config = {}

for n_estimator in n_estimators_list:
    for max_depth in max_depth_list:
        for learning_rate in learning_rate_list:

            xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimator,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective='multi:softmax',
                random_state=42
            )
            # SOLUTION: xgb.XGBClassifier(n_estimators=n_estimator, max_depth=max_depth,
            #               learning_rate=learning_rate, objective='multi:softmax', random_state=42)

            xgb_model.fit(X_train_normalized, y_train)
            xgb_preds = xgb_model.predict(X_test_normalized)
            xgb_acc = accuracy_score(y_test, xgb_preds)

            if xgb_acc < xgb_poor_acc:
                xgb_poor_acc = xgb_acc
                worst_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

            if xgb_acc > xgb_best_acc:
                xgb_best_acc = xgb_acc
                best_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

print(f"Worst Config: {worst_config}")
print(f"Best Config:  {best_config}")

# =============================================================
# FINAL OUTPUT
# =============================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Single Logistic Regression Accuracy:   {round(lr_acc, 4)}")
print(f"Bagging (Logistic Regression) Accuracy:{round(bagging_acc, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(xgb_poor_acc, 4)}")
print(f"XGBoost Best Accuracy:  {round(xgb_best_acc, 4)}")
print("=" * 60)
