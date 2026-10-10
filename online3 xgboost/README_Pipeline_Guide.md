# 📖 Online 3 - Pipeline Overview & Exam Guide
## Comparing Ensemble Learning with XGBoost

---

## 🎯 What You're Being Asked

The exam gives you a **skeleton Python file** with `# TODO` comments.
You must fill in **~7 sections** to complete a pipeline that:

1. **Loads data** from CSV
2. **Separates** features (X) and target (y)
3. **Splits** into train/test
4. **Scales** features
5. **Trains a base learner** (e.g., Logistic Regression, Decision Tree)
6. **Trains an ensemble** (Bagging / AdaBoost / Stacking) using the base learner
7. **Trains XGBoost** with multiple hyperparameter combos → finds best & worst accuracy

---

## 🔄 The Full Pipeline (Visual)

```
CSV File
  │
  ▼
┌─────────────────────┐
│ 1. pd.read_csv()    │  ← Load data into DataFrame
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. Separate X and y │  ← Drop target cols from X, create y
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. train_test_split │  ← 80% train, 20% test
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 4. StandardScaler   │  ← Normalize features
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────────────┐
│         Three paths run in parallel:             │
│                                                  │
│  Path A: Single Base Learner → accuracy          │
│  Path B: Ensemble (Bagging/Ada/Stack) → accuracy │
│  Path C: XGBoost Grid Search → best & worst acc  │
└──────────────────────────────────────────────────┘
          ▼
┌─────────────────────┐
│ Print RESULTS       │
└─────────────────────┘
```

---

## 📝 EVERY TODO EXPLAINED WITH EXACT SYNTAX

### ✅ TODO 1: Load the dataset

**What it does:** Reads a CSV file into a Pandas DataFrame

```python
df = pd.read_csv("modified_large_iris_dataset.csv")
```

That's it. One line. The filename will be given in the question.

---

### ✅ TODO 2: Separate features (X) and target (y)

**Tricky Part:** The target might be **one-hot encoded** (multiple True/False columns).

**Example DataFrame:**
```
| sepal_length | petal_width | species_0 | species_1 | species_2 |
|-------------|-------------|-----------|-----------|-----------|
| 5.1         | 0.2         | True      | False     | False     |  ← class 0
| 7.0         | 1.4         | False     | True      | False     |  ← class 1
| 6.3         | 1.8         | False     | False     | True      |  ← class 2
```

**Converting one-hot back to single label (use this style — from ahnaf.py):**
```python
# Step 1: Drop species columns to get X
X = data.drop(columns=['species_0', 'species_1', 'species_2'])

# Step 2: Get the species columns as their own dataframe
y_onehot = data[['species_0', 'species_1', 'species_2']]

# Step 3: idxmax gives the NAME of the True column per row
target_names = y_onehot.idxmax(axis=1)
# Result: Series like: 'species_0', 'species_1', 'species_0', 'species_2', ...

# Step 4: Map name → number
cols_to_label = {'species_0': 0, 'species_1': 1, 'species_2': 2}
target = target_names.map(cols_to_label)
# Result: Series like: 0, 1, 0, 2, ...
```

**Alternate (numpy one-liner — also valid):**
```python
# np.argmax returns the INDEX of the maximum value in each row
target = np.argmax(data[['species_0','species_1','species_2']].values, axis=1)
```

**If target is a normal single column:**
```python
X = data.drop(columns=['target'])
target = data['target']
```

---

### ✅ TODO 3: Train-Test Split (80/20)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Memory trick:** `X_train, X_test, y_train, y_test` — always in this order!

---

### ✅ TODO 4: Scale Features

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform on TRAIN
X_test = scaler.transform(X_test)          # ONLY transform on TEST (never fit!)
```

**Why not fit on test?** Because test data simulates unseen data. Fitting on it = data leakage.

---

### ✅ TODO 5: Train Base Learner

**Pattern is ALWAYS the same (3 lines):**
```python
model = SomeClassifier(params)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
```

**Common base learners they might give:**

| Learner | Creation Code |
|---------|--------------|
| Logistic Regression | `LogisticRegression(max_iter=1000, random_state=42)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=1, random_state=42)` |
| KNN | `KNeighborsClassifier(n_neighbors=5)` |
| SVM | `SVC(kernel='rbf', random_state=42)` |

---

### ✅ TODO 6: Ensemble Methods

#### 🅰️ BAGGING (Manual — most likely on exam)

```python
n_estimators = 10
models = []
n_samples = len(X_train_normalized)

# Step 1: Train on bootstrap samples
for i in range(n_estimators):
    X_sample, y_sample = resample(X_train_normalized, y_train,
                                  replace=True, n_samples=n_samples, random_state=i)
    boot_model = LogisticRegression(max_iter=1000, random_state=42)
    boot_model.fit(X_sample, y_sample)
    models.append(boot_model)

# Step 2: Collect predictions
predictions = []
for model in models:
    preds = model.predict(X_test_normalized)
    predictions.append(preds)

# Step 3: Majority voting
predictions  = np.array(predictions)              # shape: [10, n_test_samples]
vote_preds   = stats.mode(predictions, axis=0).mode   # most common per test sample
bagging_acc  = accuracy_score(y_test, vote_preds)
```

**`resample`:** randomly picks N rows WITH replacement — some rows repeat, some are skipped  
**`stats.mode(..., axis=0).mode`:** for each test sample (column), the most predicted class across all 10 models

#### 🅱️ ADABOOST (if asked)

```python
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)
ada_acc = accuracy_score(y_test, ada_preds)
```

#### 🅲️ STACKING (if asked)

```python
from sklearn.ensemble import StackingClassifier

stack_model = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
stack_model.fit(X_train, y_train)
stack_preds = stack_model.predict(X_test)
stack_acc = accuracy_score(y_test, stack_preds)
```

---

### ✅ TODO 7: XGBoost Hyperparameter Search

```python
n_estimators_list = [5, 20, 50]
max_depth_list = [1, 3, 5]
learning_rate_list = [0.01, 0.1, 0.3]

xgb_best_acc = 0.0
xgb_poor_acc = 1.0

for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            if acc > xgb_best_acc:
                xgb_best_acc = acc
            if acc < xgb_poor_acc:
                xgb_poor_acc = acc
```

**Key XGBoost parameters to remember:**

| Parameter | What it does | Low value | High value |
|-----------|-------------|-----------|------------|
| `n_estimators` | Number of trees | Underfit | Overfit |
| `max_depth` | Tree depth | Simple trees | Complex trees |
| `learning_rate` | Step size | Slow learning | Fast learning |

---

## ⚠️ COMMON PITFALLS

1. **`stats.mode` result** — just `.mode` is enough (from ahnaf.py), no `.flatten()` needed
2. **`scalar.transform(X_test)`** — NOT `fit_transform` on test!
3. **One-hot to label** — use `idxmax(axis=1)` then `.map(dict)` (readable) OR `np.argmax(..., axis=1)` (short)
4. **`resample` needs `random_state=i`** — use loop variable `i` so each bootstrap is different
5. **XGBoost multiclass** — add `objective='multi:softmax'` or `objective='multi:softprob'`
6. **Variable names in final print** — match exactly what the skeleton uses (`lr_acc`, `bagging_acc`, etc.)
7. **`drop(columns=[...])` vs `drop([...], axis=1)`** — both work, pick one and remember it

---

## 🧠 Ensemble Methods Comparison

```
┌──────────────┬──────────────────────────────────────────────────┐
│ BAGGING      │ Train same model on DIFFERENT bootstrap samples  │
│              │ Combine by MAJORITY VOTE                         │
│              │ Reduces VARIANCE                                 │
│              │ Models train in PARALLEL                         │
├──────────────┼──────────────────────────────────────────────────┤
│ BOOSTING     │ Train models SEQUENTIALLY                        │
│ (AdaBoost)   │ Each model focuses on PREVIOUS ERRORS            │
│              │ Reduces BIAS                                     │
│              │ Weighted voting                                  │
├──────────────┼──────────────────────────────────────────────────┤
│ STACKING     │ Train DIFFERENT types of models                  │
│              │ A meta-learner combines their predictions        │
│              │ Most flexible                                    │
├──────────────┼──────────────────────────────────────────────────┤
│ XGBOOST      │ Advanced boosting with regularization            │
│              │ Second-order gradients (Hessian)                 │
│              │ Built-in handling of missing values              │
│              │ Fastest tree-based method                        │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 💡 HOW TO FIND SYNTAX ON THE LAB COMPUTER (if you forget)

You are allowed to think of Python itself as your reference. Here are tricks that work **without internet**:

### Trick 1: `help()` in Python terminal
```python
# Open a python terminal and type:
import sklearn
help(sklearn.linear_model.LogisticRegression)   # shows all params
help(sklearn.utils.resample)                     # shows exact signature
help(scipy.stats.mode)                           # shows what it returns
```

### Trick 2: `?` in IPython / Jupyter
```python
from sklearn.utils import resample
resample?          # shows docstring instantly
```

### Trick 3: `Ctrl+Space` autocomplete in VS Code
```python
# Type the function name with ( and VS Code shows the signature
resample(          # press Ctrl+Space → see all parameters
```

### Trick 4: `dir()` to list all methods of an object
```python
import xgboost as xgb
dir(xgb)                    # list everything in xgboost
dir(xgb.XGBClassifier())    # list all methods on the model
```

### Trick 5: Read the skeleton's existing imports and print statements
The variable names in the final `print()` block tell you EXACTLY what variables you need to create. Work backwards from there.

Good luck! 🍀

