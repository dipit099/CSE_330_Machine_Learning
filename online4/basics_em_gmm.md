# EM Algorithm & Gaussian Mixture Models — Basics

---

## 1. Why GMM? What Problem Does It Solve?

K-Means forces each point into exactly one cluster (hard assignment).
But what if a data point sits between two clusters? It could partially belong to both.

**GMM** models data as coming from a **mixture of k Gaussian distributions**.
Each point gets a **probability** of belonging to each component — called **soft assignment**.

> Think of it as: "This height measurement is 80% cat, 20% dog."

---

## 2. Gaussian (Normal) Distribution

### 1D Gaussian
```
           1                   (x - mu)^2
P(x) = --------- * exp( - -------------------- )
        sigma*sqrt(2*pi)         2 * sigma^2

mu    = mean (center)
sigma = standard deviation (spread)
```

### Multivariate Gaussian (d-dimensional)
```
                    1                          1
P(x) = ------------------------- * exp( - --- (x-mu)^T * Sigma^-1 * (x-mu) )
        sqrt((2*pi)^d * |Sigma|)            2

mu    = mean vector        shape (d,)
Sigma = covariance matrix  shape (d, d)
```

---

## 3. Gaussian Mixture Model (GMM)

A GMM has **k components**. Each component i has:
- `w_i`     — mixing weight (prior probability)  →  sum of all w_i = 1
- `mu_i`    — mean vector
- `Sigma_i` — covariance matrix

The total probability of a point x:
```
P(x) = sum_{i=1}^{k}  w_i * N(x | mu_i, Sigma_i)
```

**The problem:** We don't know which component generated each point.
**The solution:** EM algorithm — iteratively estimate hidden assignments and parameters.

---

## 4. The EM Algorithm

EM alternates between two steps until **log-likelihood converges**.

### E-Step (Expectation) — "Who generated this point?"

Compute the **responsibility** `r_ij` = probability that component i generated point j:

```
r_ij  =  w_i * N(x_j | mu_i, Sigma_i)
        ---------------------------------------
         sum_{i'} w_{i'} * N(x_j | mu_{i'}, Sigma_{i'})
```

- Result: matrix R of shape (N, k)
- Each row sums to 1
- This is Bayes' rule: posterior = (prior * likelihood) / evidence

### M-Step (Maximization) — "Update parameters using soft assignments"

For each component i:

```
n_i   = sum_j r_ij                                   ← effective count

mu_i  = (1/n_i) * sum_j  r_ij * x_j                 ← weighted mean

Sigma_i = (1/n_i) * sum_j  r_ij * (x_j - mu_i)(x_j - mu_i)^T   ← weighted cov

w_i   = n_i / N                                      ← new mixing weight
```

### Log-Likelihood (Convergence Check)

```
L = sum_{j=1}^{N}  log( sum_{i=1}^{k}  w_i * N(x_j | mu_i, Sigma_i) )

Stop when:  |L_new - L_old| < epsilon   (e.g. epsilon = 1e-4)
```

Log-likelihood only increases at each EM iteration (guaranteed).

---

## 5. EM Algorithm — Full Picture

```
INITIALIZE:
  mu_i    = k random data points
  Sigma_i = identity matrix for each i
  w_i     = 1/k for each i

REPEAT:
  E-step:   compute R (N x k responsibility matrix)
  M-step:   update mu, Sigma, w using R
  Compute:  log-likelihood L
  Check:    |L_new - L_old| < epsilon → STOP

OUTPUT:
  labels = argmax over columns of R   (hard assignment from soft probs)
```

---

## 6. Key Properties

| Property | Detail |
|---|---|
| Soft assignment | Each point gets probability of belonging to each cluster |
| Clusters shape | Can be elliptical (unlike K-Means spheres) |
| Convergence | Log-likelihood guaranteed to increase each iteration |
| Local optima | Can get stuck — run multiple times (`n_init`) |
| Saddle points | Optimization is not globally convex |
| Choose k | Use **AIC** or **BIC** (lower = better) |

### AIC vs BIC
```
AIC = 2 * n_params - 2 * log-likelihood        (penalizes complexity less)
BIC = log(N) * n_params - 2 * log-likelihood   (penalizes complexity more)

Both: lower = better
BIC tends to prefer simpler models (smaller k)
```

---

## 7. Covariance Types in sklearn

```python
GaussianMixture(covariance_type='full')    # each component has its own full Sigma (most flexible)
GaussianMixture(covariance_type='tied')    # all components share one Sigma
GaussianMixture(covariance_type='diag')    # diagonal covariance (no feature correlation)
GaussianMixture(covariance_type='spherical') # single variance per component (like K-Means)
```

Default: `'full'` — use this unless told otherwise.

---

## 8. sklearn API

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(
    n_components    = k,        # number of Gaussian components
    covariance_type = 'full',   # type of covariance matrix
    n_init          = 3,        # run EM n_init times, keep best
    random_state    = 42
)

gm.fit(X_scaled)

gm.predict(X_scaled)           # hard labels          shape (N,)
gm.predict_proba(X_scaled)     # soft responsibilities shape (N, k)
gm.score(X_scaled)             # mean log-likelihood per sample  (higher=better)
gm.score_samples(X_scaled)     # log-likelihood per sample       shape (N,)

gm.aic(X_scaled)               # AIC score (lower=better)
gm.bic(X_scaled)               # BIC score (lower=better)

gm.means_                      # shape (k, d)
gm.covariances_                # shape (k, d, d)  for 'full'
gm.weights_                    # shape (k,)
gm.converged_                  # True if EM converged
gm.n_iter_                     # number of EM iterations used
```

---

## 9. From-Scratch EM Code Structure (what exam asks)

```python
from scipy.stats import multivariate_normal

def e_step(X, means, covs, weights, k):
    N = len(X)
    R = np.zeros((N, k))
    for i in range(k):
        R[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covs[i])
    R /= R.sum(axis=1, keepdims=True)    # normalize rows
    return R

def m_step(X, R, k):
    N, d = X.shape
    means, covs, weights = [], [], []
    for i in range(k):
        r_i   = R[:, i]
        n_i   = r_i.sum()
        mu_i  = (r_i[:, None] * X).sum(axis=0) / n_i
        diff  = X - mu_i
        sig_i = (r_i[:, None] * diff).T @ diff / n_i + 1e-6*np.eye(d)
        means.append(mu_i); covs.append(sig_i); weights.append(n_i / N)
    return np.array(means), covs, np.array(weights)

def log_likelihood(X, means, covs, weights, k):
    ll = 0
    for j in range(len(X)):
        mix = sum(weights[i] * multivariate_normal.pdf(X[j], means[i], covs[i]) for i in range(k))
        ll += np.log(mix + 1e-300)
    return ll

# Main loop
prev_ll = -np.inf
for iteration in range(max_iters):
    R            = e_step(X, means, covs, weights, k)
    means, covs, weights = m_step(X, R, k)
    ll           = log_likelihood(X, means, covs, weights, k)
    if abs(ll - prev_ll) < tol:
        break
    prev_ll = ll
```

---

## 10. GMM vs K-Means — When to Use Which

| Situation | Use |
|---|---|
| Want simple, fast clustering | K-Means |
| Want probabilities (soft assignments) | GMM |
| Clusters are elongated or non-spherical | GMM |
| Need to pick k automatically | Both (Silhouette for KMeans, BIC for GMM) |
| Exam asks to implement from scratch | Know both E-step and M-step formulas |

---

## 11. Common Mistakes

- Forgetting to **normalize** responsibilities in E-step (rows must sum to 1)
- Using `log(0)` — always add `1e-300` inside log
- Forgetting **regularization** (`+1e-6 * I`) in covariance update → singular matrix crash
- Using silhouette/inertia for GMM — use **AIC/BIC** instead
- Not checking `gm.converged_` — if False, increase `max_iter`
