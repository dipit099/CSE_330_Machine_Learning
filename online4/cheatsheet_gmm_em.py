"""
===============================================================
 EXAM CHEATSHEET: GMM + EM ALGORITHM
 Topics: Unsupervised Learning, EM
===============================================================
 KEY FORMULAS:
   Gaussian PDF:
     N(x | mu, sigma^2) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2 / 2*sigma^2)

   Mixture:
     P(x) = sum_i  w_i * N(x | mu_i, Sigma_i)

   E-step (soft assignment / responsibility):
     r_ij = w_i * N(x_j | mu_i, Sigma_i)
     r_ij /= sum_i r_ij            (normalize so each row sums to 1)

   M-step (parameter update):
     n_i   = sum_j r_ij                          (effective count)
     mu_i  = (1/n_i) * sum_j r_ij * x_j          (weighted mean)
     Sig_i = (1/n_i) * sum_j r_ij*(x_j-mu_i)*(x_j-mu_i)^T  (weighted cov)
     w_i   = n_i / N                              (mixing weight)

   Log-likelihood (convergence check):
     L = sum_j log( sum_i  w_i * N(x_j | mu_i, Sigma_i) )
     Stop when |L_new - L_old| < epsilon
===============================================================
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(42)

# ================================================================
# SECTION 1: DATA
# ================================================================

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================
# SECTION 2: SKLEARN GaussianMixture (THE QUICK WAY)
# ================================================================

k = 3

# TODO: create and fit GMM
gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, random_state=42)
gmm.fit(X_scaled)

# Key attributes:
labels_hard = gmm.predict(X_scaled)          # hard cluster assignment  shape (N,)
labels_soft = gmm.predict_proba(X_scaled)    # soft (responsibilities)  shape (N, k)
log_likelihood = gmm.score(X_scaled)         # mean log-likelihood per sample

print(f"Hard labels (first 5): {labels_hard[:5]}")
print(f"Responsibilities (first row): {labels_soft[0].round(4)}")
print(f"Log-likelihood per sample: {log_likelihood:.4f}")

# Fitted params (useful to inspect):
print(f"Means:\n{gmm.means_}")
print(f"Weights: {gmm.weights_}")
# gmm.covariances_  — shape (k, d, d)

# ================================================================
# SECTION 3: CHOOSE k — AIC / BIC (use these for GMM, NOT inertia)
# ================================================================

# AIC = lower is better   BIC = lower is better
aics = []
bics = []
k_range = range(1, 8)

# TODO: loop, fit, record AIC/BIC
for k_val in k_range:
    gm = GaussianMixture(n_components=k_val, n_init=3, random_state=42)
    gm.fit(X_scaled)
    aics.append(gm.aic(X_scaled))
    bics.append(gm.bic(X_scaled))
    print(f"  k={k_val}  AIC={gm.aic(X_scaled):.2f}  BIC={gm.bic(X_scaled):.2f}")

best_k_aic = k_range[np.argmin(aics)]
best_k_bic = k_range[np.argmin(bics)]
print(f"\nBest k by AIC: {best_k_aic}   Best k by BIC: {best_k_bic}")

plt.figure()
plt.plot(k_range, aics, 'r-o', label='AIC')
plt.plot(k_range, bics, 'b-o', label='BIC')
plt.xlabel("k"); plt.ylabel("Score (lower = better)")
plt.legend(); plt.title("AIC / BIC for GMM")
plt.tight_layout(); plt.savefig("gmm_aic_bic.png"); plt.close()

# ================================================================
# SECTION 4: GMM-EM FROM SCRATCH (what exam will ask!)
# ================================================================

def gaussian_pdf(x, mean, cov):
    """Multivariate Gaussian pdf using scipy (handles numerical stability)."""
    # TODO: return N(x | mean, cov)
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

def initialize_params(X, k):
    """Random initialization of GMM parameters."""
    N, d = X.shape
    # TODO: pick k random data points as initial means
    idx = np.random.choice(N, k, replace=False)
    means   = X[idx].copy()                          # shape (k, d)
    covs    = [np.eye(d) for _ in range(k)]          # identity covariances
    weights = np.ones(k) / k                          # uniform mixing weights
    return means, covs, weights

def e_step(X, means, covs, weights, k):
    """
    E-Step: compute responsibility matrix R.
    R[j, i] = probability that point j belongs to component i.
    """
    N = len(X)
    R = np.zeros((N, k))   # responsibilities

    # TODO: for each component i, compute weighted gaussian pdf
    for i in range(k):
        R[:, i] = weights[i] * gaussian_pdf(X, means[i], covs[i])

    # TODO: normalize each row so responsibilities sum to 1
    row_sums = R.sum(axis=1, keepdims=True)
    R /= row_sums   # shape (N, k)
    return R

def m_step(X, R, k):
    """
    M-Step: update parameters using responsibility matrix R.
    """
    N, d = X.shape
    means   = np.zeros((k, d))
    covs    = []
    weights = np.zeros(k)

    for i in range(k):
        r_i = R[:, i]           # responsibilities for component i, shape (N,)

        # TODO: effective count
        n_i = r_i.sum()

        # TODO: new mean  (weighted average)
        means[i] = (r_i[:, np.newaxis] * X).sum(axis=0) / n_i

        # TODO: new covariance  (weighted outer product)
        diff = X - means[i]     # shape (N, d)
        cov_i = (r_i[:, np.newaxis] * diff).T @ diff / n_i
        cov_i += 1e-6 * np.eye(d)   # numerical stability (regularization)
        covs.append(cov_i)

        # TODO: new weight
        weights[i] = n_i / N

    return means, covs, weights

def compute_log_likelihood(X, means, covs, weights, k):
    """L = sum_j log( sum_i w_i * N(x_j | mu_i, Sigma_i) )"""
    N = len(X)
    ll = 0.0
    for j in range(N):
        # TODO: sum contributions from all components
        mix = sum(weights[i] * gaussian_pdf(X[j], means[i], covs[i]) for i in range(k))
        ll += np.log(mix + 1e-300)   # add tiny value to avoid log(0)
    return ll

def gmm_em_scratch(X, k, max_iters=100, tol=1e-4):
    """Full GMM-EM from scratch."""
    means, covs, weights = initialize_params(X, k)
    log_likelihoods = []

    prev_ll = -np.inf

    for iteration in range(max_iters):
        # TODO: E-step
        R = e_step(X, means, covs, weights, k)

        # TODO: M-step
        means, covs, weights = m_step(X, R, k)

        # TODO: compute log-likelihood
        ll = compute_log_likelihood(X, means, covs, weights, k)
        log_likelihoods.append(ll)

        print(f"  Iter {iteration+1:3d}  log-likelihood = {ll:.4f}")

        # TODO: check convergence
        if abs(ll - prev_ll) < tol:
            print(f"  Converged at iteration {iteration+1}")
            break
        prev_ll = ll

    # Hard labels from final responsibilities
    R_final = e_step(X, means, covs, weights, k)
    labels = np.argmax(R_final, axis=1)

    return labels, means, covs, weights, log_likelihoods

print("\n--- GMM-EM from scratch ---")
labels_scratch, means_s, covs_s, weights_s, lls = gmm_em_scratch(X_scaled, k=3)

plt.figure()
plt.plot(lls)
plt.xlabel("Iteration"); plt.ylabel("Log-likelihood")
plt.title("EM Convergence")
plt.tight_layout(); plt.savefig("em_convergence.png"); plt.close()
print("EM convergence plot saved → em_convergence.png")

# ================================================================
# SECTION 5: COMPARE SKLEARN vs SCRATCH
# ================================================================

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_hard,    cmap='tab10', s=10)
plt.title("sklearn GMM")
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scratch, cmap='tab10', s=10)
plt.title("GMM-EM from scratch")
plt.tight_layout(); plt.savefig("gmm_compare.png"); plt.close()
print("Comparison plot saved → gmm_compare.png")

# ================================================================
# QUICK REFERENCE (copy-paste syntax)
# ================================================================
# from sklearn.mixture import GaussianMixture
#
# gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, random_state=42)
# gmm.fit(X_scaled)
# labels     = gmm.predict(X_scaled)          # hard labels
# probs      = gmm.predict_proba(X_scaled)    # soft responsibilities
# ll_score   = gmm.score(X_scaled)            # mean log-likelihood
# aic        = gmm.aic(X_scaled)
# bic        = gmm.bic(X_scaled)
# gmm.means_         # (k, d)
# gmm.covariances_   # (k, d, d)
# gmm.weights_       # (k,)
