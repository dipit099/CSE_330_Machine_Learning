# 🎯 ML Online Exam Cheatsheet
## NumPy, Pandas & Matplotlib - Quick Reference

---

# 📚 TABLE OF CONTENTS
1. [Pandas - Data Loading & EDA](#1-pandas---data-loading--eda)
2. [Pandas - Data Cleaning](#2-pandas---data-cleaning)
3. [NumPy - Array Operations](#3-numpy---array-operations)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Training Pipeline (NumPy Only)](#5-training-pipeline-numpy-only)
6. [Matplotlib - Visualization](#6-matplotlib---visualization)
7. [Common TODO Patterns](#7-common-todo-patterns)
8. [Quick Syntax Comparison Table](#8-quick-syntax-comparison-table)

---

# 1. PANDAS - DATA LOADING & EDA

## 📂 Loading Data
```python
import pandas as pd

# Load CSV file
df = pd.read_csv("filename.csv")

# Load with options
df = pd.read_csv("filename.csv", header=0)        # first row as header
df = pd.read_csv("filename.csv", index_col=0)     # first col as index
df = pd.read_csv("filename.csv", na_values=['?']) # treat '?' as NaN
```

## 🔍 Basic Exploration
```python
df.shape              # (rows, columns) - tuple
df.head()             # first 5 rows
df.head(10)           # first 10 rows
df.tail()             # last 5 rows
df.info()             # column names, types, non-null counts
df.describe()         # statistics (mean, std, min, max, quartiles)
df.columns            # list of column names
df.dtypes             # data type of each column
len(df)               # number of rows
```

## 🔍 conditions
```python
df[(df['Age'] > 20) & (df['CGPA'] > 3.6)]
df[(df['Dept'] == 'CSE') | (df['CGPA'] >= 3.8)]
df[df['Age'] > 21] # filter rows where Age > 21
df[~(df['Dept'] == 'CSE')]
# grade assign
# df.loc[condition, column] = value
df.loc[df['CGPA'] >= 3.75, 'Grade'] = 'A' # assign 'A' in  Grade column where CGPA >= 3.75


# LOC:
df.loc[101] # row with index 101
df.loc[100:110] # rows from index 100 to 110
df.loc[:, ['Name', 'CGPA']] # all rows, specific columns
# ILOC: 
df.iloc[0] # first row
df.iloc[0:10] # first 10 rows
df.iloc[:, [1, 3]] # all rows, specific columns by index

```

## 📊 Value Inspection
```python
df['column'].unique()           # unique values in column
df['column'].nunique()          # count of unique values
df['column'].value_counts()     # frequency count of each value
df['target'].map({0: 'A', 1: 'B'})  # map values to new values
```

---

# 2. PANDAS - DATA CLEANING
##
```python
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')
```

## ❓ Handling NULL Values
```python
# CHECK nulls
df.isnull()                    # True/False for each cell
df.isnull().sum()              # null count per column
df.isnull().sum().sum()        # total null count (single number)
df.isna().sum()                # same as isnull()

# FILL nulls
df.fillna(0)                   # fill all nulls with 0
df.fillna(value=0)             # same as above
df['col'].fillna(0)            # fill nulls in specific column
df.fillna(df.mean())           # fill with column means
df['col'].fillna(df['col'].mean())  # fill column with its mean

# wit mean, median, mode
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['CGPA'].fillna(df['CGPA'].median(), inplace=True)
df['Dept'].fillna(df['Dept'].mode()[0], inplace=True)


# DROP nulls
df.dropna()                    # drop rows with ANY null
df.dropna(axis=0)              # drop rows (axis=0 is rows)
df.dropna(axis=1)              # drop columns with nulls
df.dropna(how='all')           # drop rows where ALL values are null
```

## 🔁 Handling Duplicates
```python
# CHECK duplicates
df.duplicated()                # True/False for each row
df.duplicated().sum()          # count of duplicate rows

# DROP duplicates
df.drop_duplicates()           # drop duplicate rows
df.drop_duplicates(keep='first')   # keep first occurrence
df.drop_duplicates(keep='last')    # keep last occurrence
df.drop_duplicates(subset=['col']) # check duplicates in specific column
```

## ✂️ Selecting & Dropping Columns
```python
# SELECT columns
df['column']                   # single column (Series)
df[['col1', 'col2']]           # multiple columns (DataFrame)
df.iloc[:, :-1]                # all columns except last
df.iloc[:, -1]                 # last column only

# DROP columns
df.drop('column', axis=1)              # drop single column
df.drop(['col1', 'col2'], axis=1)      # drop multiple columns
df.drop(['target', 'species'], axis=1) # example from demo
```

## 🎯 Extracting X (features) and y (target)
```python
# Method 1: Using column names
X = df.drop('target', axis=1)          # all columns except 'target'
y = df['target']                        # target column

df.iloc[rows, columns]
# Method 2: Using iloc (index-based)
X = df.iloc[:, :-1]                    # all rows, all columns except last
y = df.iloc[:, -1]                     # all rows, last column only

# Method 3: Drop multiple columns
X = df.drop(['target', 'species'], axis=1)
y = df['target']

# Convert to NumPy arrays
X = df.iloc[:, :-1].values             # .values converts to numpy array
y = df.iloc[:, -1].values              # .values converts to numpy array
## here (4,) no Row,coln -> after using reshape -> (4, 1)
y = y.reshape(-1, 1)                   # reshape to column vector
| অংশ  | মানে                                |
| ---- | ----------------------------------- |
| `-1` | rows সংখ্যা **নিজে নিজে হিসাব করো** |
| `1`  | ১টা column বানাও                    |

```

---

# 3. NUMPY - ARRAY OPERATIONS

## 📐 Creating Arrays
```python
import numpy as np

np.array([1, 2, 3])            # from list
np.zeros((3, 4))               # 3x4 array of zeros
np.ones((3, 4))                # 3x4 array of ones
np.eye(3)                      # 3x3 identity matrix
np.random.randn(3, 4)          # random normal distribution
np.random.rand(3, 4)           # random uniform [0, 1)
np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)           # 5 evenly spaced values from 0 to 1
np.argmin(a)   # index of minimum
np.argmax(a)   # index of maximum
np.mean(a, axis=0)  # column-wise
np.mean(a, axis=1)  # row-wise
np.abs(a)
np.power(a, 2)
np.sort(a)
np.argsort(a). #Returns the indices that would sort the array. #a[np.argsort(a)]
X = np.random.rand(100, 10) #X.shape   # (100, 10). #row,col
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0) ## z score # 


np.unique(a)
```

## 📏 Array Properties
```python
arr.shape                      # (rows, cols) tuple
arr.ndim                       # number of dimensions
arr.size                       # total number of elements
arr.dtype                      # data type
len(arr)                       # length of first dimension
```

## 🔄 Reshaping
```python
arr.reshape(3, 4)              # reshape to 3x4
arr.reshape(-1, 1)             # column vector (auto-calculate rows)
arr.reshape(1, -1)             # row vector (auto-calculate cols)
arr.flatten()                  # 1D array
arr.ravel()                    # 1D array (view, faster)
arr.T                          # transpose
```

## ➕ Mathematical Operations
```python
# Element-wise operations
a + b                          # addition
a - b                          # subtraction
a * b                          # element-wise multiplication
a / b                          # division
a ** 2                         # power
np.exp(a)                      # exponential e^a
np.log(a)                      # natural log
np.sqrt(a)                     # square root

# Matrix operations
a @ b                          # matrix multiplication (preferred)
np.dot(a, b)                   # matrix multiplication
a.dot(b)                       # matrix multiplication
```

## 📊 Statistical Functions
```python
arr.mean()                     # mean of all elements
arr.mean(axis=0)               # mean along columns
arr.mean(axis=1)               # mean along rows
arr.std()                      # standard deviation
arr.std(axis=0)                # std along columns
arr.sum()                      # sum all
arr.sum(axis=0)                # sum along columns
arr.min(), arr.max()           # min, max
arr.argmin(), arr.argmax()     # index of min, max
np.mean(arr)                   # same as arr.mean()
```

## 🔗 Stacking & Concatenating
```python
np.vstack([a, b])              # vertical stack (row-wise)
np.hstack([a, b])              # horizontal stack (column-wise)
np.concatenate([a, b], axis=0) # along rows
np.concatenate([a, b], axis=1) # along columns
```

## 🎲 Indexing & Slicing
```python
arr[0]                         # first row
arr[0, 1]                      # element at row 0, col 1
arr[:, 0]                      # first column (all rows)
arr[0, :]                      # first row (all columns)
arr[0:3, :]                    # rows 0-2, all columns
arr[i:i+batch_size]            # batch slicing
```

---

# 4. DATA PREPROCESSING

## 📏 Normalization (Z-score / StandardScaler)
```python
# Formula: X_norm = (X - mean) / std

# NumPy way (column-wise normalization)
X_mean = X.mean(axis=0)                    # mean of each column
X_std = X.std(axis=0)                      # std of each column
X_normalized = (X - X_mean) / (X_std + 1e-8)  # add small value to avoid /0

# Shorthand
X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
```

## 📏 Min-Max Normalization
```python
# Formula: X_norm = (X - min) / (max - min)
# Result: values between 0 and 1

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
```

## ✂️ Train-Test Split (Manual with NumPy)
```python
# Simple split
n_samples = len(X)
n_train = int(0.8 * n_samples)             # 80% train

indices = np.arange(n_samples)
np.random.shuffle(indices)                  # shuffle indices

train_idx = indices[:n_train]
test_idx = indices[n_train:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

---

# 5. TRAINING PIPELINE (NUMPY ONLY)

## 🔄 Mini-Batch Processing
```python
batch_size = 32
n_samples = X.shape[0]

for i in range(0, n_samples, batch_size):
    X_batch = X[i:i+batch_size]            # slice batch
    y_batch = y[i:i+batch_size]
    
    # Process batch...
```

## 🧠 Forward Pass (Linear Model)
```python
# Initialize weights
n_features = X.shape[1]
W = np.zeros((n_features, 1))              # weight matrix
b = 0.0                                     # bias

# Forward pass: y_pred = X @ W + b
predictions = X @ W + b                     # matrix multiplication
# or
predictions = np.dot(X, W) + b
# or  
predictions = X.dot(W) + b
```

## 🎯 Common Activation Functions
```python
# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ReLU
def relu(z):
    return np.maximum(0, z)

# Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

## 📉 Loss Functions
```python
# Mean Squared Error (MSE) - Regression
mse = np.mean((y_pred - y_true) ** 2)

# Binary Cross-Entropy - Binary Classification
bce = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
```

## 🔄 Gradient Descent Update
```python
learning_rate = 0.01
n_samples = X.shape[0]

# Compute gradients
dW = (1/n_samples) * X.T @ (y_pred - y)    # gradient w.r.t W
db = (1/n_samples) * np.sum(y_pred - y)    # gradient w.r.t b

# Update weights
W = W - learning_rate * dW
b = b - learning_rate * db
```

## 📊 Accuracy Calculation
```python
# For binary classification (threshold = 0.5)
y_pred_class = (predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_true)

# Or
accuracy = np.sum(y_pred_class == y_true) / len(y_true)

# For multiclass (one-hot or probabilities)
y_pred_class = np.argmax(predictions, axis=1)
y_true_class = np.argmax(y_true, axis=1)
accuracy = np.mean(y_pred_class == y_true_class)
```

---

# 6. MATPLOTLIB - VISUALIZATION

## 📈 Basic Plotting
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot(x, y)
plt.plot(x, y, label='line1')              # with label for legend

# Scatter plot
plt.scatter(x, y)
plt.scatter(x, y, c=colors, cmap='viridis') # colored by category

# Show plot
plt.show()
```

## 🎨 Plot Customization
```python
plt.figure(figsize=(10, 6))                # set figure size
plt.title('My Plot')                       # title
plt.xlabel('X axis')                       # x-axis label
plt.ylabel('Y axis')                       # y-axis label
plt.legend()                               # show legend
plt.grid(True)                             # show grid
plt.tight_layout()                         # adjust spacing
```

## 📊 Subplots
```python
# Method 1
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Plot 1')

# Method 2
plt.subplot(2, 2, 1)                       # row=2, col=2, index=1
plt.plot(x, y)
```

## 📊 Histogram
```python
plt.hist(data, bins=20)                    # histogram with 20 bins
plt.hist(data, bins=20, edgecolor='black') # with edge color
```

## 🔥 Heatmap (for confusion matrix)
```python
import seaborn as sns

# Using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Using matplotlib only
plt.imshow(confusion_matrix, cmap='Blues')
plt.colorbar()
```

---

# 7. COMMON TODO PATTERNS

## Pattern 1: Load CSV and Fill Nulls
```python
# TODO: Use pandas to load the file
df = pd.read_csv(filename)

# TODO: Print the number of null values in each column
print(df.isnull().sum())

# TODO: Fill nulls with fill_value
df = df.fillna(fill_value)

# TODO: Verify null values after filling
print(df.isnull().sum())
```

## Pattern 2: Extract Features and Target
```python
# TODO: Separate features (X) and target (y)
X = df.iloc[:, :-1].values     # all columns except last
y = df.iloc[:, -1].values      # last column

# or with column names:
X = df.drop('target', axis=1)
y = df['target']
```

## Pattern 3: Normalize Data
```python
# TODO: Normalize the features
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
```

## Pattern 4: Forward Pass
```python
# TODO: Compute predictions using matrix multiplication
predictions = X @ W + b
# or
predictions = np.dot(X, W) + b
```

## Pattern 5: Calculate Accuracy
```python
# TODO: Calculate accuracy
y_pred_class = (predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred_class == y)
```

## Pattern 6: Mini-Batch Loop
```python
# TODO: Process data in batches
for i in range(0, n_samples, batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    # forward pass
    preds = X_batch @ W + b
```

---

# 8. QUICK SYNTAX COMPARISON TABLE

## Pandas vs NumPy Conversion

| Task | Pandas | NumPy |
|------|--------|-------|
| Get values | `df.values` | already array |
| Shape | `df.shape` | `arr.shape` |
| Mean | `df.mean()` | `arr.mean()` |
| Column mean | `df['col'].mean()` | `arr.mean(axis=0)` |
| Row mean | `df.mean(axis=1)` | `arr.mean(axis=1)` |
| Sum | `df.sum()` | `arr.sum()` |
| Min/Max | `df.min()`, `df.max()` | `arr.min()`, `arr.max()` |

## Common Operations Quick Reference

| Task | Code |
|------|------|
| Load CSV | `df = pd.read_csv('file.csv')` |
| Check nulls | `df.isnull().sum()` |
| Fill nulls | `df.fillna(0)` |
| Drop nulls | `df.dropna()` |
| Check duplicates | `df.duplicated().sum()` |
| Drop duplicates | `df.drop_duplicates()` |
| Drop column | `df.drop('col', axis=1)` |
| Get X, y | `X = df.iloc[:, :-1].values` |
| | `y = df.iloc[:, -1].values` |
| Normalize | `X = (X - X.mean(axis=0)) / X.std(axis=0)` |
| Forward pass | `y_pred = X @ W + b` |
| Accuracy | `np.mean(y_pred == y)` |
| Batch slice | `X[i:i+batch_size]` |

## Axis Reference (IMPORTANT!)

| axis | Meaning | Example |
|------|---------|---------|
| `axis=0` | Along rows (down) | `df.drop('row', axis=0)` |
| `axis=1` | Along columns (across) | `df.drop('col', axis=1)` |
| `axis=0` in stats | Compute per column | `mean(axis=0)` → one value per column |
| `axis=1` in stats | Compute per row | `mean(axis=1)` → one value per row |

---

# 🚀 FULL PIPELINE TEMPLATE

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. LOAD DATA ==========
df = pd.read_csv("data.csv")
print("Shape:", df.shape)
print(df.head())

# ========== 2. CLEAN DATA ==========
# Check and fill nulls
print("Nulls:", df.isnull().sum())
df = df.fillna(0)  # or df.fillna(df.mean())

# Check and drop duplicates
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()

# ========== 3. PREPARE X, y ==========
X = df.iloc[:, :-1].values      # features
y = df.iloc[:, -1].values.reshape(-1, 1)  # target

# ========== 4. NORMALIZE ==========
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# ========== 5. INITIALIZE WEIGHTS ==========
n_samples, n_features = X.shape
W = np.zeros((n_features, 1))
b = 0.0

# ========== 6. TRAINING LOOP ==========
learning_rate = 0.01
epochs = 100
batch_size = 32

for epoch in range(epochs):
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        # Forward pass
        y_pred = X_batch @ W + b
        
        # Compute gradients
        m = len(X_batch)
        dW = (1/m) * X_batch.T @ (y_pred - y_batch)
        db = (1/m) * np.sum(y_pred - y_batch)
        
        # Update weights
        W = W - learning_rate * dW
        b = b - learning_rate * db

# ========== 7. EVALUATE ==========
predictions = X @ W + b
y_pred_class = (predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred_class == y)
print(f"Accuracy: {accuracy:.4f}")
```

---

# ⚡ QUICK RECALL TIPS

1. **Load → Clean → Split → Normalize → Train → Evaluate**

2. **Pandas for data loading/cleaning, NumPy for computation**

3. **`axis=0` = columns, `axis=1` = rows** (think: drop column → axis=1)

4. **Normalization formula**: `(X - mean) / std`

5. **Forward pass**: `X @ W + b` or `np.dot(X, W) + b`

6. **Batch loop**: `for i in range(0, n, batch_size): X[i:i+batch_size]`

7. **Accuracy**: `np.mean(predictions == y_true)`

8. **Always add `1e-8` to denominators to avoid division by zero!**

---

Good luck with your exam! 🍀
