# 🚀 Online 3 Cheatsheet — Quick Exam Reference
# Ensemble Learning + XGBoost
# All syntax matches ahnaf.py style (step-by-step, easy to remember)
# ================================================================

# ----------------------------------------------------------------
# IMPORTS (already given in exam, don't change)
# ----------------------------------------------------------------
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

# Only needed if AdaBoost / Stacking is asked:
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier


# ================================================================
# 1. LOAD DATA
# ================================================================
data = pd.read_csv("filename.csv")


# ================================================================
# 2. SEPARATE X AND TARGET
# ================================================================

# --- Case A: one-hot encoded target (most likely in exam) ---
X = data.drop(columns=['species_0', 'species_1', 'species_2'])

y_onehot = data[['species_0', 'species_1', 'species_2']]
target_names = y_onehot.idxmax(axis=1)       # gives 'species_0', 'species_1', etc.
cols_to_label = {'species_0': 0, 'species_1': 1, 'species_2': 2}
target = target_names.map(cols_to_label)     # converts to 0, 1, 2

# --- Case B: normal single target column ---
# X = data.drop(columns=['target'])
# target = data['target']


# ================================================================
# 3. TRAIN-TEST SPLIT (80/20)
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, target, train_size=0.8, random_state=42
)


# ================================================================
# 4. SCALE FEATURES
# ================================================================
scalar = StandardScaler()
X_train_normalized = scalar.fit_transform(X_train)   # fit + scale on train
X_test_normalized  = scalar.transform(X_test)        # scale only on test!


# ================================================================
# 5. BASE LEARNERS — pick whichever is asked
# ================================================================
# The 3-line pattern is the SAME for every model:
#   model = SomeClassifier(...)
#   model.fit(X_train_normalized, y_train)
#   preds = model.predict(X_test_normalized)
#   acc = accuracy_score(y_test, preds)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_normalized, y_train)
lr_preds = lr_model.predict(X_test_normalized)
lr_acc = accuracy_score(y_test, lr_preds)

# Decision Tree stump (max_depth=1 = very weak learner)
# dt_model = DecisionTreeClassifier(max_depth=1, random_state=42)

# KNN
# knn_model = KNeighborsClassifier(n_neighbors=5)

# SVM
# svm_model = SVC(kernel='rbf', random_state=42)


# ================================================================
# 6A. ENSEMBLE — BAGGING (manual, most likely in exam)
# ================================================================
n_estimators = 10
models = []

n_samples = len(X_train_normalized)

for i in range(n_estimators):
    # resample = random pick WITH replacement (bootstrap sample)
    X_sample, y_sample = resample(X_train_normalized, y_train,
                                  replace=True, n_samples=n_samples, random_state=i)
    boot_model = LogisticRegression(max_iter=1000, random_state=42)
    boot_model.fit(X_sample, y_sample)
    models.append(boot_model)

predictions = []
for model in models:
    preds = model.predict(X_test_normalized)
    predictions.append(preds)

predictions = np.array(predictions)       # shape: [n_estimators, n_test_samples]
vote_preds  = stats.mode(predictions, axis=0).mode   # majority vote
bagging_acc = accuracy_score(y_test, vote_preds)


# ================================================================
# 6B. ENSEMBLE — ADABOOST (if asked)
# ================================================================
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_model.fit(X_train_normalized, y_train)
ada_preds = ada_model.predict(X_test_normalized)
ada_acc = accuracy_score(y_test, ada_preds)


# ================================================================
# 6C. ENSEMBLE — STACKING (if asked)
# ================================================================
stack_model = StackingClassifier(
    estimators=[
        ('dt',  DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
stack_model.fit(X_train_normalized, y_train)
stack_preds = stack_model.predict(X_test_normalized)
stack_acc = accuracy_score(y_test, stack_preds)


# ================================================================
# 7. XGBOOST — HYPERPARAMETER GRID SEARCH
# ================================================================
n_estimators_list  = [5, 20, 50]
max_depth_list     = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

xgb_poor_acc = 1.0
xgb_best_acc = 0.0
worst_config = {}
best_config  = {}

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
            xgb_model.fit(X_train_normalized, y_train)
            xgb_preds = xgb_model.predict(X_test_normalized)
            xgb_acc = accuracy_score(y_test, xgb_preds)

            if xgb_acc < xgb_poor_acc:
                xgb_poor_acc = xgb_acc
                worst_config = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}

            if xgb_acc > xgb_best_acc:
                xgb_best_acc = xgb_acc
                best_config  = {'n_estimators': n_estimator, 'max_depth': max_depth, 'learning_rate': learning_rate}


# ================================================================
# PRINT RESULTS
# ================================================================
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Base Learner (LR) Accuracy:   {round(lr_acc, 4)}")
print(f"Bagging Accuracy:             {round(bagging_acc, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(xgb_poor_acc, 4)}  Config: {worst_config}")
print(f"XGBoost Best Accuracy:  {round(xgb_best_acc, 4)}  Config: {best_config}")
print("=" * 60)
