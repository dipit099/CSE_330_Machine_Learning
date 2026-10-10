"""
===============================================================
 PRACTICE QUESTION (Exam-style with TODOs)
 Simulates what your teacher typically asks in online tests.
===============================================================
 TASK:
   Given a dataset:
   1. Scale the features
   2. Apply PCA to reduce dimensions (keep 95% variance)
   3. Cluster with KMeans — find best k via elbow + silhouette
   4. Cluster with GMM — find best k via AIC/BIC
   5. Compare results
===============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------

# TODO: load from CSV
# df = pd.read_csv("data.csv")
# X = df.values

# Using synthetic data for practice:
X, y_true = make_blobs(n_samples=400, n_features=6, centers=4, cluster_std=1.2, random_state=42)

print(f"Data shape: {X.shape}")

# ----------------------------------------------------------------
# STEP 1: Scale the features
# ----------------------------------------------------------------

# TODO: scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------
# STEP 2: PCA — keep 95% variance
# ----------------------------------------------------------------

# TODO: fit PCA with enough components to keep 95% explained variance
pca = PCA(n_components=0.95, random_state=42)   # pass float → auto pick n
X_pca = pca.fit_transform(X_scaled)

n_comps = pca.n_components_
print(f"\nPCA: kept {n_comps} components for 95% variance")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.round(4)}")
print(f"Cumulative: {pca.explained_variance_ratio_.cumsum().round(4)}")

# ----------------------------------------------------------------
# STEP 3: KMeans — find best k
# ----------------------------------------------------------------

inertias    = []
sil_scores  = []
k_range     = range(2, 8)

# TODO: loop over k values, fit KMeans, record inertia and silhouette
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, km.labels_))

best_k_km = k_range[np.argmax(sil_scores)]
print(f"\nKMeans — best k (silhouette): {best_k_km}")

# TODO: fit final KMeans with best k
km_final = KMeans(n_clusters=best_k_km, n_init=10, random_state=42)
km_final.fit(X_pca)
km_labels = km_final.labels_

print(f"KMeans inertia: {km_final.inertia_:.4f}")
print(f"KMeans silhouette: {silhouette_score(X_pca, km_labels):.4f}")

# ----------------------------------------------------------------
# STEP 4: GMM — find best k via BIC
# ----------------------------------------------------------------

bics = []

# TODO: loop over k values, fit GaussianMixture, record BIC
for k in k_range:
    gm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, random_state=42)
    gm.fit(X_pca)
    bics.append(gm.bic(X_pca))

best_k_gm = list(k_range)[np.argmin(bics)]
print(f"\nGMM — best k (BIC): {best_k_gm}")

# TODO: fit final GMM with best k
gm_final = GaussianMixture(n_components=best_k_gm, covariance_type='full', n_init=3, random_state=42)
gm_final.fit(X_pca)
gm_labels     = gm_final.predict(X_pca)
gm_probs      = gm_final.predict_proba(X_pca)
gm_ll         = gm_final.score(X_pca)

print(f"GMM log-likelihood (per sample): {gm_ll:.4f}")
print(f"GMM BIC: {gm_final.bic(X_pca):.4f}")
print(f"GMM silhouette: {silhouette_score(X_pca, gm_labels):.4f}")

# ----------------------------------------------------------------
# STEP 5: Print final results
# ----------------------------------------------------------------

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"PCA components kept     : {n_comps}")
print(f"KMeans best k           : {best_k_km}")
print(f"KMeans silhouette score : {silhouette_score(X_pca, km_labels):.4f}")
print(f"GMM best k              : {best_k_gm}")
print(f"GMM log-likelihood      : {gm_ll:.4f}")
print("="*60)

# ----------------------------------------------------------------
# STEP 6: Plots
# ----------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Elbow
axes[0].plot(k_range, inertias, 'bo-')
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow (KMeans)")

# Silhouette vs k
axes[1].plot(k_range, sil_scores, 'go-')
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette (KMeans)")

# BIC vs k
axes[2].plot(k_range, bics, 'rs-')
axes[2].set_xlabel("k"); axes[2].set_ylabel("BIC")
axes[2].set_title("BIC (GMM)")

plt.tight_layout()
plt.savefig("practice_results.png"); plt.close()
print("\nPlots saved → practice_results.png")
