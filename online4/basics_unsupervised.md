# Unsupervised Learning — Basics

> No labels. The algorithm finds structure in data by itself.

---

## 1. What is Unsupervised Learning?

| Supervised | Unsupervised |
|---|---|
| Has labels y | No labels |
| Learn X → y mapping | Find hidden structure in X |
| Accuracy is measurable | Harder to evaluate |
| Examples: classification, regression | Examples: clustering, dimensionality reduction |

---

## 2. K-Means Clustering

### Core Idea
Partition N data points into k clusters such that each point belongs to the cluster with the **nearest centroid** (mean).

### Algorithm — Step by Step

```
1. INIT:     Randomly pick k data points as initial centroids (mu_1 ... mu_k)

2. ASSIGN:   For each point x_j:
               label[j] = argmin_i  ||x_j - mu_i||^2      ← nearest centroid

3. UPDATE:   For each cluster i:
               mu_i = mean of all x_j where label[j] == i  ← new centroid

4. REPEAT:   Go to step 2 until centroids stop moving (convergence)
```

### Key Formula

```
Inertia (WCSS) = sum over all points of  ||x_j - mu_{label[j]}||^2

Lower inertia = tighter clusters (but always decreases as k increases!)
```

### How to Pick k?

**Elbow Method**
- Plot inertia vs k
- Look for the "elbow" — the bend where adding more clusters gives diminishing returns

**Silhouette Score**
```
For each point x:
  a = mean distance to other points in same cluster
  b = mean distance to points in nearest OTHER cluster
  s = (b - a) / max(a, b)

Range: [-1, 1]   →   1 = perfect, 0 = overlapping, -1 = wrong cluster
```
- Pick k with highest average silhouette score

### Properties
- **Hard assignment** — each point belongs to exactly 1 cluster
- **Sensitive to initialization** — use `n_init=10` to run multiple times
- **Sensitive to scale** — always StandardScale before KMeans
- **Assumes spherical clusters** — bad for elongated or unequal-size clusters
- **Local optima** — not guaranteed to find global optimum

### sklearn API
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=k, n_init=10, random_state=42)
km.fit(X_scaled)

km.labels_             # cluster id for each point   shape (N,)
km.cluster_centers_    # centroid coordinates        shape (k, features)
km.inertia_            # WCSS value
km.predict(X_new)      # assign new points

sil = silhouette_score(X_scaled, km.labels_)
```

---

## 3. PCA — Principal Component Analysis

### Core Idea
Find the directions (**principal components**) along which the data has the **most variance**, then project the data onto the top-k of these directions to reduce dimensions.

### Why Use PCA?
- Reduce features before clustering (speeds up, reduces noise)
- Visualize high-dimensional data in 2D/3D
- Remove correlated features
- Noise filtering

### Algorithm — Step by Step

```
1. CENTER:   X = X - mean(X, axis=0)

2. COVARIANCE MATRIX:
             C = X.T @ X / (N - 1)     shape: (d, d)

3. EIGENDECOMPOSE:
             eigenvalues, eigenvectors = np.linalg.eigh(C)
             (eigh is for symmetric matrices — more stable than eig)

4. SORT:     Sort eigenvectors by descending eigenvalue

5. SELECT:   Pick top-k eigenvectors → these are the Principal Components
             components = eigenvectors[:, :k].T    shape: (k, d)

6. PROJECT:  X_reduced = X_centered @ components.T   shape: (N, k)
```

### Key Concepts

| Term | Meaning |
|---|---|
| Principal Component | Eigenvector of covariance matrix (direction of max variance) |
| Explained Variance Ratio | % of total variance captured by each PC |
| Cumulative Variance | Sum of EVR — tells you how many PCs to keep |
| Scree Plot | Plot of EVR vs component index — look for elbow |

### Choosing n_components

```python
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Keep enough components for 95% variance:
n = np.argmax(cumvar >= 0.95) + 1

# OR — pass directly:
pca = PCA(n_components=0.95)   # sklearn auto picks n
```

### sklearn API
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)              # or n_components=0.95
X_reduced = pca.fit_transform(X_scaled)

pca.explained_variance_ratio_          # array, per component
pca.explained_variance_ratio_.cumsum() # cumulative
pca.components_                        # eigenvectors shape (k, d)
pca.n_components_                      # how many were auto-selected
pca.singular_values_                   # singular values
```

### From Scratch
```python
def pca_scratch(X, n_components):
    mean = X.mean(axis=0)
    X_c  = X - mean
    C    = X_c.T @ X_c / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order      = np.argsort(eigenvalues)[::-1]        # sort descending
    components = eigenvectors[:, order][:, :n_components].T
    return X_c @ components.T                         # projected data
```

---

## 4. K-Means vs GMM Quick Comparison

| Feature | K-Means | GMM |
|---|---|---|
| Assignment | Hard (1 cluster only) | Soft (probabilities) |
| Shape assumption | Spherical clusters | Any elliptical shape |
| Output | Labels | Labels + probabilities |
| Choose k | Elbow + Silhouette | AIC / BIC |
| Speed | Fast | Slower |
| Initialization sensitivity | High | High |

---

## 5. Typical Exam Pipeline

```
Load data → StandardScale → (optional PCA) → Cluster → Evaluate

Step 1:  scaler = StandardScaler(); X_s = scaler.fit_transform(X)
Step 2:  pca = PCA(n_components=0.95); X_p = pca.fit_transform(X_s)
Step 3:  km  = KMeans(n_clusters=k, n_init=10, random_state=42); km.fit(X_p)
Step 4:  silhouette_score(X_p, km.labels_)
```

---

## 6. Common Mistakes

- Forgetting to `StandardScaler` before KMeans/PCA
- Using `fit_transform` on test data (should be just `transform`)
- For clustering there is no train/test split — fit on all X
- Using inertia/silhouette for GMM (use AIC/BIC instead)
- Using `np.linalg.eig` instead of `np.linalg.eigh` for symmetric matrices
