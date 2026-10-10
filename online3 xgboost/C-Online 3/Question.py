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

df = pd.read_csv("modified_large_iris_dataset.csv")

# TODO: separate features (X) and target (y)
# The species columns are one-hot encoded: species_0, species_1, species_2
# Convert them back to a single target column

# help(df.drop)

x = df.drop(columns=["species_0","species_1","species_2"])

y=[]
for i in range(len(df)):
    if df['species_0'].iloc[i] == 1 :
        y.append(0)
    elif df['species_1'].iloc[i] == 1 :
        y.append(1)
    elif df['species_2'].iloc[i] == 1 :
        y.append(2)


y = pd.Series(y)
print(type(x))
print(type(y))
print(x.head(2))
print(y.head(2))



# TODO: train-test split (80%-20%)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,stratify=y, random_state=42)
print(f"X_train.shape = {x_train.shape}")
print(f"X_test.shape = {x_test.shape}")

# TODO: scale the features if needed


scalar = StandardScaler()
x_train_norm = scalar.fit_transform(x_train)
x_test_norm = scalar.transform(x_test)
print("done3")
print(type(x_test_norm))

print("done4")
print(x_test_norm[:2])
# -------------------- Single Logistic Regression --------------------

# TODO: train a single Logistic Regression (max_iter=1000, random_state=42) and compute lr_acc

simple_Reg = LogisticRegression(max_iter=1000, random_state=42)
simple_Reg.fit(x_train_norm,y_train)
simple_Reg_preds=simple_Reg.predict(x_test_norm)

lr_acc =  accuracy_score(y_test,simple_Reg_preds)

# -------------------- Bagging with Logistic Regression --------------------

n_estimators = 10
models = []
n_samples= len(x_train_norm)

# TODO: train n_estimators Logistic Regression (max_iter=1000, random_state=42) models on bootstrap samples

for i in range(n_estimators):
    x_sample,y_sample = resample(x_train_norm,y_train,n_samples=n_samples,replace=True)
    demo_model = LogisticRegression(max_iter=1000, random_state=42)
    demo_model.fit(x_sample,y_sample)
    
    models.append(demo_model)


print("done5")
# TODO: collect predictions from all models

predictions = []
for demo_model in models:
    preds = demo_model.predict(x_test_norm)
    predictions.append(preds)

# TODO: perform majority voting and compute bagging_acc

print(predictions[:2])
vote_preds = stats.mode(predictions,axis=0).mode
print(vote_preds[:2])
bagging_acs = accuracy_score(y_test, vote_preds)

print(f"bagging acc {bagging_acs}")
# -------------------- XGBoost --------------------

# TODO: experiment XGBoost with different combinations of hyperparameters (n_estimators, max_depth, and learning_rate).

n_estimators_list = [5, 20, 50]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

# TODO: train and predict with each combination
# Find the worst(Lowest possible XGBoost accuracy) and best(Highest possible XGBoost accuracy) XGBoost configurations

best_score= 0.0
bad_score=1.0
for n_estimator in n_estimators_list:
    for depth in max_depth_list:
        for rate in learning_rate_list:
            xgb_model = xgb.XGBClassifier(
                n_estimators= n_estimator,
                max_depth = depth,
                learning_rate = rate,
                objective = "multi:softmax"
        
            )
            xgb_model.fit(x_train_norm,y_train)
            xgb_preds = xgb_model.predict(x_test_norm)
            score = accuracy_score(y_test,xgb_preds)
            if score >= best_score :
                best_score=score
            if score<= bad_score:
                bad_score=score


# -------------------- Final Output --------------------

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Single Logistic Regression Accuracy: {round(lr_acc, 4)}")
print(f"Bagging (Logistic Regression) Accuracy: {round(bagging_acs, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(bad_score, 4)}")
print(f"XGBoost Best Accuracy: {round(best_score, 4)}")
print("=" * 60)