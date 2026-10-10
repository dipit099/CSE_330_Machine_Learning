# EXAM CHEATSHEET — Unsupervised Learning & EM
> Read this file right before the exam. All syntax, solutions, and gotchas in one place.

---

## IMPORTS (copy at top of every exam file)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)
```

---

## PIPELINE (most common exam pattern)

```
Load CSV → Scale → (PCA if asked) → Cluster → Evaluate → Print results
```

```python
# 1. Load
df = pd.read_csv("data.csv")
X  = df.values                         # or df.drop(columns=[...]).values

# 2. Scale  ← ALWAYS do this first
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)     # NO train/test split for clustering

# 3. PCA (if asked to reduce dims)
pca       = PCA(n_components=0.95)     # keeps enough for 95% variance
X_reduced = pca.fit_transform(X_scaled)
print(f"PCA kept {pca.n_components_} components")
print(f"Explained variance: {pca.explained_variance_ratio_.round(4)}")

# 4a. KMeans
km = KMeans(n_clusters=k, n_init=10, random_state=42)
km.fit(X_reduced)
km_labels   = km.labels_
km_inertia  = km.inertia_
km_sil      = silhouette_score(X_reduced, km_labels)

# 4b. GMM
gm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, random_state=42)
gm.fit(X_reduced)
gm_labels   = gm.predict(X_reduced)
gm_probs    = gm.predict_proba(X_reduced)    # soft assignments
gm_ll       = gm.score(X_reduced)            # mean log-likelihood
gm_bic      = gm.bic(X_reduced)
```

---

## FIND BEST k

### KMeans — Elbow + Silhouette

```python
inertias   = []
sil_scores = []
k_range    = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"Best k (silhouette): {best_k}")
```

### GMM — AIC / BIC

```python
aics = []
bics = []

for k in range(1, 10):
    gm = GaussianMixture(n_components=k, n_init=3, random_state=42)
    gm.fit(X_scaled)
    aics.append(gm.aic(X_scaled))
    bics.append(gm.bic(X_scaled))

best_k_aic = np.argmin(aics) + 1   # +1 because range starts at 1
best_k_bic = np.argmin(bics) + 1
print(f"Best k AIC: {best_k_aic}   Best k BIC: {best_k_bic}")
```

---

## K-MEANS — FULL SOLUTION (from scratch)

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(X, centroids):
    labels = []
    for x in X:
        dists = [euclidean_distance(x, c) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels)

def update_centroids(X, labels, k):
    centroids = []
    for i in range(k):
        pts = X[labels == i]
        centroids.append(pts.mean(axis=0) if len(pts) > 0 else X[np.random.randint(len(X))])
    return np.array(centroids)

def kmeans_scratch(X, k, max_iters=100, tol=1e-4):
    idx       = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()

    for it in range(max_iters):
        labels        = assign_clusters(X, centroids)       # E-like step
        new_centroids = update_centroids(X, labels, k)      # M-like step
        shift         = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids     = new_centroids
        if shift < tol:
            print(f"Converged at iter {it+1}"); break

    inertia = sum(euclidean_distance(X[j], centroids[labels[j]])**2 for j in range(len(X)))
    return labels, centroids, inertia

# Usage
labels, centroids, inertia = kmeans_scratch(X_scaled, k=3)
```

---

## PCA — FULL SOLUTION (from scratch)

```python
def pca_scratch(X, n_components):
    mean      = X.mean(axis=0)
    X_c       = X - mean                             # center
    C         = X_c.T @ X_c / (len(X) - 1)          # covariance matrix
    evals, evecs = np.linalg.eigh(C)                 # eigendecompose (eigh for symmetric)
    order     = np.argsort(evals)[::-1]              # sort descending
    evecs     = evecs[:, order]
    comps     = evecs[:, :n_components].T            # top-k eigenvectors, shape (k, d)
    X_reduced = X_c @ comps.T                        # project, shape (N, k)

    evr = evals[order] / evals.sum()
    print(f"Explained variance ratio: {evr[:n_components].round(4)}")
    print(f"Cumulative: {evr[:n_components].cumsum().round(4)}")
    return X_reduced, comps

# Usage
X_2d, components = pca_scratch(X_scaled, n_components=2)
```

---

## GMM-EM — FULL SOLUTION (from scratch)

### E-Step
```python
def e_step(X, means, covs, weights, k):
    N = len(X)
    R = np.zeros((N, k))
    for i in range(k):
        # weighted Gaussian pdf for each component
        R[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covs[i])
    # normalize: each row must sum to 1 (responsibilities)
    R /= R.sum(axis=1, keepdims=True)
    return R   # shape (N, k)
```

### M-Step
```python
def m_step(X, R, k):
    N, d      = X.shape
    new_means = np.zeros((k, d))
    new_covs  = []
    new_weights = np.zeros(k)

    for i in range(k):
        r_i = R[:, i]                               # responsibilities for component i
        n_i = r_i.sum()                             # effective count

        new_means[i]  = (r_i[:, None] * X).sum(axis=0) / n_i       # weighted mean
        diff          = X - new_means[i]
        cov_i         = (r_i[:, None] * diff).T @ diff / n_i        # weighted covariance
        cov_i        += 1e-6 * np.eye(d)                            # regularize!
        new_covs.append(cov_i)
        new_weights[i] = n_i / N                                    # new mixing weight

    return new_means, new_covs, new_weights
```

### Log-Likelihood
```python
def log_likelihood(X, means, covs, weights, k):
    ll = 0.0
    for j in range(len(X)):
        mix = sum(weights[i] * multivariate_normal.pdf(X[j], means[i], covs[i])
                  for i in range(k))
        ll += np.log(mix + 1e-300)      # 1e-300 avoids log(0) crash
    return ll
```

### Full EM Loop
```python
def gmm_em(X, k, max_iters=100, tol=1e-4):
    N, d = X.shape

    # Initialize
    idx      = np.random.choice(N, k, replace=False)
    means    = X[idx].copy()
    covs     = [np.eye(d) for _ in range(k)]
    weights  = np.ones(k) / k
    prev_ll  = -np.inf

    for it in range(max_iters):
        R              = e_step(X, means, covs, weights, k)       # E-step
        means, covs, weights = m_step(X, R, k)                    # M-step
        ll             = log_likelihood(X, means, covs, weights, k)
        print(f"Iter {it+1}: log-likelihood = {ll:.4f}")

        if abs(ll - prev_ll) < tol:                               # convergence
            print(f"Converged at iter {it+1}"); break
        prev_ll = ll

    R_final = e_step(X, means, covs, weights, k)
    labels  = np.argmax(R_final, axis=1)                          # hard labels
    return labels, means, covs, weights

# Usage
labels, means, covs, weights = gmm_em(X_scaled, k=3)
```

---

## SKLEARN ATTRIBUTE QUICK LOOKUP

### KMeans
| Attribute / Method | What it gives |
|---|---|
| `km.labels_` | Cluster id for each point  `(N,)` |
| `km.cluster_centers_` | Centroid coords  `(k, features)` |
| `km.inertia_` | WCSS — sum of squared distances (lower=better) |
| `km.predict(X_new)` | Assign new points to clusters |
| `km.fit_predict(X)` | Fit + return labels in one call |

### GaussianMixture
| Attribute / Method | What it gives |
|---|---|
| `gm.predict(X)` | Hard labels  `(N,)` |
| `gm.predict_proba(X)` | Soft responsibilities  `(N, k)` |
| `gm.score(X)` | Mean log-likelihood per sample (higher=better) |
| `gm.score_samples(X)` | Log-likelihood per sample  `(N,)` |
| `gm.aic(X)` | AIC score (lower=better) |
| `gm.bic(X)` | BIC score (lower=better) |
| `gm.means_` | Fitted means  `(k, d)` |
| `gm.covariances_` | Fitted covariances  `(k, d, d)` |
| `gm.weights_` | Mixing weights  `(k,)` |
| `gm.converged_` | True if EM converged |
| `gm.n_iter_` | Number of EM iterations used |

### PCA
| Attribute / Method | What it gives |
|---|---|
| `pca.fit_transform(X)` | Fit and project  `(N, k)` |
| `pca.transform(X_new)` | Project new data (after fitting) |
| `pca.explained_variance_ratio_` | % variance per component |
| `pca.explained_variance_ratio_.cumsum()` | Cumulative variance |
| `pca.components_` | Eigenvectors  `(k, d)` |
| `pca.n_components_` | How many components were selected |

---

## EVALUATION METRICS SUMMARY

| Metric | For | Formula / Range | Better |
|---|---|---|---|
| Inertia (WCSS) | KMeans | sum of squared distances to centroid | Lower |
| Silhouette Score | KMeans | `(b-a)/max(a,b)`, range [-1,1] | Higher (close to 1) |
| AIC | GMM | `2*params - 2*LL` | Lower |
| BIC | GMM | `log(N)*params - 2*LL` | Lower |
| Log-likelihood | GMM/EM | `sum log P(x_j)` | Higher (less negative) |

---

## GOTCHAS — DON'T MESS THESE UP

```
1. ALWAYS StandardScaler before KMeans/PCA
2. For clustering: fit on ALL data, no train/test split
3. GMM k selection: AIC or BIC, NOT inertia/silhouette
4. E-step normalization: rows of R must sum to 1
5. Regularize covariance: += 1e-6 * np.eye(d)  to avoid singular matrix
6. log(0) crash: use np.log(mix + 1e-300)
7. eigh not eig: use np.linalg.eigh for symmetric matrices (covariance)
8. PCA n_components=0.95 (float): auto-pick number of components
9. Silhouette score: needs k >= 2
```

---

## FORMULA SHEET

```
Gaussian PDF (1D):
  N(x|mu,sigma) = (1/sigma*sqrt(2*pi)) * exp(-(x-mu)^2 / 2*sigma^2)

Mixture probability:
  P(x) = sum_i  w_i * N(x | mu_i, Sigma_i)

E-step (responsibility):
  r_ij = w_i * N(x_j|mu_i,Sigma_i) / sum_{i'} w_{i'} * N(x_j|mu_{i'},Sigma_{i'})

M-step:
  n_i     = sum_j r_ij
  mu_i    = (1/n_i) * sum_j r_ij * x_j
  Sigma_i = (1/n_i) * sum_j r_ij * (x_j-mu_i)(x_j-mu_i)^T
  w_i     = n_i / N

Log-likelihood:
  L = sum_j log( sum_i w_i * N(x_j|mu_i,Sigma_i) )
  Stop when |L_new - L_old| < epsilon

KMeans objective (minimize):
  J = sum_j  ||x_j - mu_{label[j]}||^2    (= inertia)

Silhouette:
  s = (b - a) / max(a, b)
  a = mean dist to same-cluster points
  b = mean dist to nearest other cluster
```
