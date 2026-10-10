# PRACTICE 2 — AdaBoost & Stacking (alternative ensemble methods)
# Style: step-by-step, simple, easy to remember
# Dataset: sklearn iris (no CSV needed for this practice)
#
# Use this if your exam asks for AdaBoost or Stacking instead of manual Bagging
# =============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
import xgboost as xgb
import random
from scipy import stats

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# =============================================================
# STEP 1: Load & prepare data
# =============================================================
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data   # shape: (150, 4)
y = iris.target # values: 0, 1, 2

# TODO: split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
# SOLUTION: train_test_split(X, y, train_size=0.8, random_state=42)

# TODO: scale features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
# SOLUTION: scalar.fit_transform(X_train) then scalar.transform(X_test)

# =============================================================
# STEP 2: Single Decision Tree (the "base learner")
# =============================================================
# max_depth=1 makes it a "decision stump" — very weak learner

# TODO: create, train, predict, compute accuracy
dt_model = DecisionTreeClassifier(max_depth=1, random_state=42)
# SOLUTION: DecisionTreeClassifier(max_depth=1, random_state=42)

dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

print(f"Single Decision Tree Accuracy: {dt_acc:.4f}")

# =============================================================
# STEP 3: AdaBoost with Decision Tree
# =============================================================
# AdaBoost trains models one after another
# Each new model focuses on the MISTAKES of the previous one
# Final answer = weighted vote from all models

# TODO: create AdaBoostClassifier, train, predict, accuracy

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # base learner
    n_estimators=50,                                 # how many weak learners
    learning_rate=1.0,                               # weight of each learner
    random_state=42
)
# SOLUTION: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
#               n_estimators=50, learning_rate=1.0, random_state=42)

ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)
ada_acc = accuracy_score(y_test, ada_preds)

print(f"AdaBoost Accuracy: {ada_acc:.4f}")

# =============================================================
# STEP 4: Stacking
# =============================================================
# Stacking trains DIFFERENT model types as base learners
# Then a "meta-learner" (final_estimator) learns from their outputs

# TODO: create StackingClassifier, train, predict, accuracy

stack_model = StackingClassifier(
    estimators=[
        ('dt',  DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
# SOLUTION: StackingClassifier(estimators=[('dt',...),('knn',...),('svm',...)],
#               final_estimator=LogisticRegression(...), cv=5)

stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_test)
stack_acc = accuracy_score(y_test, stack_preds)

print(f"Stacking Accuracy: {stack_acc:.4f}")

# =============================================================
# STEP 5: Manual Bagging (same as Practice 1, for comparison)
# =============================================================
n_estimators = 10
models = []

for i in range(n_estimators):
    X_sample, y_sample = resample(X_train, y_train, replace=True, n_samples=len(X_train), random_state=i)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_sample, y_sample)
    models.append(model)

predictions = []
for model in models:
    preds = model.predict(X_test)
    predictions.append(preds)

predictions = np.array(predictions)
vote_preds = stats.mode(predictions, axis=0).mode
bag_dt_acc = accuracy_score(y_test, vote_preds)

print(f"Manual Bagging (DT) Accuracy: {bag_dt_acc:.4f}")

# =============================================================
# STEP 6: XGBoost Hyperparameter Search
# =============================================================
n_estimators_list = [10, 50, 100]
max_depth_list = [2, 4, 6]
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
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_test)
            xgb_acc = accuracy_score(y_test, xgb_preds)

            if xgb_acc < xgb_poor_acc:
                xgb_poor_acc = xgb_acc
                worst_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

            if xgb_acc > xgb_best_acc:
                xgb_best_acc = xgb_acc
                best_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

print(f"\nXGBoost Worst: {xgb_poor_acc:.4f}  Config: {worst_config}")
print(f"XGBoost Best:  {xgb_best_acc:.4f}  Config: {best_config}")

# =============================================================
# FINAL OUTPUT
# =============================================================
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"Single Decision Tree:   {dt_acc:.4f}")
print(f"AdaBoost:               {ada_acc:.4f}")
print(f"Stacking:               {stack_acc:.4f}")
print(f"Manual Bagging (DT):    {bag_dt_acc:.4f}")
print(f"XGBoost Best:           {xgb_best_acc:.4f}")
print(f"XGBoost Worst:          {xgb_poor_acc:.4f}")
print("=" * 60)
