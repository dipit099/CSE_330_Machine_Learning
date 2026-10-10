"""
===============================================================
 EXAM CHEATSHEET: PCA (Principal Component Analysis)
 Topics: Unsupervised Learning — Dimensionality Reduction
===============================================================
 KEY IDEAS:
   - Finds directions (eigenvectors) of max variance in data
   - Projects data onto top-k eigenvectors → reduces dimensions
   - Eigendecomposition of covariance matrix:  Cov = (1/N) X^T X
   - Explained variance ratio tells how much info each PC keeps

 STEPS (from scratch):
   1. Center the data:  X = X - mean(X)
   2. Compute cov matrix:  C = X.T @ X / (N-1)
   3. Eigendecompose:  eigenvalues, eigenvectors = np.linalg.eigh(C)
   4. Sort by descending eigenvalue
   5. Pick top k eigenvectors (principal components)
   6. Project:  X_reduced = X @ components[:k].T
===============================================================
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

np.random.seed(42)

# ================================================================
# SECTION 1: DATA
# ================================================================

iris = load_iris()
X    = iris.data      # shape (150, 4)
y    = iris.target    # used only for coloring plots

# TODO: scale BEFORE PCA (PCA is variance-based, scale matters!)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================
# SECTION 2: SKLEARN PCA (THE QUICK WAY)
# ================================================================

# TODO: fit PCA — reduce to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)     # shape (N, 2)

# Key attributes:
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", pca.explained_variance_ratio_.cumsum())
print("Components (eigenvectors) shape:", pca.components_.shape)  # (n_components, n_features)
print("Singular values:", pca.singular_values_)

# ================================================================
# SECTION 3: HOW MANY COMPONENTS? (Cumulative variance plot)
# ================================================================

pca_full = PCA()                     # fit all components
pca_full.fit(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# TODO: pick n_components where cumulative variance >= threshold (e.g. 95%)
threshold = 0.95
n_components_needed = np.argmax(cumvar >= threshold) + 1
print(f"\nComponents needed for {threshold*100:.0f}% variance: {n_components_needed}")

plt.figure()
plt.plot(range(1, len(cumvar)+1), cumvar, 'bo-')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100:.0f}% threshold')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.legend()
plt.tight_layout(); plt.savefig("pca_scree.png"); plt.close()
print("Scree plot saved → pca_scree.png")

# ================================================================
# SECTION 4: VISUALIZE 2D PROJECTION
# ================================================================

plt.figure()
for label, name in enumerate(iris.target_names):
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, s=20)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Iris dataset — PCA 2D projection")
plt.legend()
plt.tight_layout(); plt.savefig("pca_iris.png"); plt.close()
print("PCA 2D plot saved → pca_iris.png")

# ================================================================
# SECTION 5: PCA FROM SCRATCH
# ================================================================

def pca_scratch(X, n_components):
    """PCA from scratch using eigendecomposition."""
    N, d = X.shape

    # TODO: Step 1 — center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # TODO: Step 2 — covariance matrix
    cov = X_centered.T @ X_centered / (N - 1)   # shape (d, d)

    # TODO: Step 3 — eigendecomposition
    # np.linalg.eigh is for symmetric matrices (more stable than eig)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # TODO: Step 4 — sort by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]   # columns are eigenvectors

    # TODO: Step 5 — pick top n_components
    components = eigenvectors[:, :n_components].T   # shape (n_components, d)

    # TODO: Step 6 — project data
    X_reduced = X_centered @ components.T           # shape (N, n_components)

    # Explained variance ratio
    evr = eigenvalues / eigenvalues.sum()
    print(f"\n[Scratch] Explained variance ratio: {evr[:n_components].round(4)}")
    print(f"[Scratch] Cumulative:               {evr[:n_components].cumsum().round(4)}")

    return X_reduced, components, evr

X_pca_scratch, comps, evr = pca_scratch(X_scaled, n_components=2)

# The sign of PCs can flip — compare magnitudes, not direction
print(f"\nSklearn  PC1 (abs): {np.abs(pca.components_[0]).round(4)}")
print(f"Scratch  PC1 (abs): {np.abs(comps[0]).round(4)}")

# ================================================================
# SECTION 6: PCA + KMEANS PIPELINE (common exam combo)
# ================================================================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# TODO: reduce dims first, then cluster
pca_3 = PCA(n_components=3)
X_3d  = pca_3.fit_transform(X_scaled)

km = KMeans(n_clusters=3, n_init=10, random_state=42)
km.fit(X_3d)
labels = km.labels_
sil    = silhouette_score(X_3d, labels)
print(f"\nKMeans on PCA(3D) — silhouette: {sil:.4f}  inertia: {km.inertia_:.4f}")

# ================================================================
# QUICK REFERENCE (copy-paste syntax)
# ================================================================
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# pca = PCA(n_components=k)         # or PCA(n_components=0.95) → auto pick
# X_reduced = pca.fit_transform(X_scaled)
# pca.explained_variance_ratio_     # per-component
# pca.explained_variance_ratio_.cumsum()  # cumulative
# pca.components_                   # eigenvectors (k, d)
# pca.singular_values_
