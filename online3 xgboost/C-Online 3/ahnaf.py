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

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# TODO: load the dataset

data = pd.read_csv("modified_large_iris_dataset.csv")
# print(data.head())
# print(data.info())

# TODO: separate features (X) and target (y)
# The species columns are one-hot encoded: species_0, species_1, species_2
# Convert them back to a single target column

X = data.drop(columns=['species_0', 'species_1', 'species_2'])
y = data[['species_0', 'species_1', 'species_2']]

# print(X.head())
# print(X.info())

# print(y.head())
# print(y.info())

# def set_target_column(row):
#     if row.species_0 == True:
#         row['label'] = 0
#     elif row.species_1 == True:
#         row['label'] = 1
#     else:
#         row['label'] = 2
    
#     return row

# y_converted = y.apply(set_target_column, axis='columns')
#Iterate row by row.
# For each row (a pandas Series), pass it to set_target_column.
# Collect the returned result into a new DataFrame
# target = y_converted['label']

# ==========
# print(target.head())
# print(target.info())

target_names = y.idxmax(axis=1)
cols_to_label = {'species_0': 0, 'species_1': 1, 'species_2': 2}
target = target_names.map(cols_to_label)
target.name = "label"

# print(target.head())
# print(target.info())

# TODO: train-test split (80%-20%)

X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8, stratify=target)

# TODO: scale the features if needed

scalar = StandardScaler()
X_train_normalized = scalar.fit_transform(X_train)
X_test_normalized = scalar.transform(X_test)

# -------------------- Single Logistic Regression --------------------

# TODO: train a single Logistic Regression (max_iter=1000, random_state=42) and compute lr_acc

simple_reg = LogisticRegression(max_iter=1000, random_state=42)
simple_reg.fit(X_train_normalized, y_train)
simple_reg_preds = simple_reg.predict(X_test_normalized)
lr_acc = accuracy_score(y_test, simple_reg_preds)

# -------------------- Bagging with Logistic Regression --------------------

n_estimators = 10
models = []

# TODO: train n_estimators Logistic Regression (max_iter=1000, random_state=42) models on bootstrap samples

n_samples = len(X_train_normalized)

for i in range(n_estimators):
    X_sample, y_sample = resample(X_train_normalized, y_train, replace=True, n_samples=n_samples)
    boot_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    boot_reg_model.fit(X_sample, y_sample)
    models.append(boot_reg_model)

# TODO: collect predictions from all models

predictions = []

for model in models:
    preds = model.predict(X_test_normalized)
    predictions.append(preds)

# TODO: perform majority voting and compute bagging_acc

vote_preds = stats.mode(predictions, axis=0).mode

# help(stats.mode)

bagging_acc = accuracy_score(y_test, vote_preds)

# -------------------- XGBoost --------------------

# TODO: experiment XGBoost with different combinations of hyperparameters (n_estimators, max_depth, and learning_rate).

n_estimators_list = [5, 20, 50]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

# TODO: train and predict with each combination
# Find the worst(Lowest possible XGBoost accuracy) and best(Highest possible XGBoost accuracy) XGBoost configurations

xgb_poor_acc = 1.0
xgb_best_acc = 0.0

for n_estimator in n_estimators_list:
    for max_depth in max_depth_list:
        for learning_rate in learning_rate_list:
            xgb_model = xgb.XGBClassifier(
                objective = 'multi:softmax',
                n_estimators = n_estimator,
                learning_rate = learning_rate,
                max_depth = max_depth
            )
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
print(f"Best Config: {best_config}")

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