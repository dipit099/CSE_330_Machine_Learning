"""
===============================================================
 EXAM CHEATSHEET: K-MEANS CLUSTERING
 Topics: Unsupervised Learning
===============================================================
 KEY FORMULA:
   - Assign:  label[j] = argmin_i  ||x_j - mu_i||^2
   - Update:  mu_i = mean of all x_j assigned to cluster i
   - Inertia: sum of ||x_j - mu_{label[j]}||^2  (WCSS, lower is better)
   - Silhouette score: (b - a) / max(a, b)  in [-1, 1]  (higher is better)
===============================================================
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(42)

# ================================================================
# SECTION 1: GENERATE / LOAD DATA
# ================================================================

# Synthetic blobs (easy to visualize)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# TODO: load real data
# df = pd.read_csv("data.csv")
# X = df.drop(columns=["label"]).values   # features only — no labels in unsupervised!

# ================================================================
# SECTION 2: PREPROCESS (ALWAYS SCALE FOR KMEANS)
# ================================================================

# TODO: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # fit+transform on ALL data (no train/test split needed)

# ================================================================
# SECTION 3: SKLEARN KMEANS (THE QUICK WAY)
# ================================================================

k = 4

# TODO: create and fit KMeans
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Key attributes you must know:
labels      = kmeans.labels_          # cluster id for each sample   shape (N,)
centroids   = kmeans.cluster_centers_ # centroid coords              shape (k, features)
inertia     = kmeans.inertia_         # WCSS (within-cluster sum of squares)

print(f"Labels (first 10): {labels[:10]}")
print(f"Centroids shape  : {centroids.shape}")
print(f"Inertia (WCSS)   : {inertia:.4f}")

# TODO: predict cluster for new points
# new_point = scaler.transform([[5.0, 3.0]])
# pred = kmeans.predict(new_point)

# ================================================================
# SECTION 4: CHOOSE OPTIMAL K — ELBOW METHOD
# ================================================================

inertias = []
k_range  = range(1, 10)

# TODO: loop over k, fit KMeans, record inertia
for k_val in k_range:
    km = KMeans(n_clusters=k_val, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Elbow is where the curve bends — pick that k
plt.figure()
plt.plot(k_range, inertias, 'bo-')
plt.xlabel("k"); plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.tight_layout(); plt.savefig("elbow.png"); plt.close()
print("Elbow plot saved → elbow.png")

# ================================================================
# SECTION 5: SILHOUETTE SCORE (to validate k)
# ================================================================

sil_scores = []

# TODO: silhouette score needs k >= 2
for k_val in range(2, 10):
    km = KMeans(n_clusters=k_val, n_init=10, random_state=42)
    km.fit(X_scaled)
    score = silhouette_score(X_scaled, km.labels_)
    sil_scores.append(score)
    print(f"  k={k_val}  silhouette={score:.4f}")

best_k = np.argmax(sil_scores) + 2   # +2 because range starts at 2
print(f"\nBest k by silhouette: {best_k}")

# ================================================================
# SECTION 6: K-MEANS FROM SCRATCH (what exam may ask)
# ================================================================

def euclidean_distance(a, b):
    # TODO: ||a - b||_2
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(X, centroids):
    # TODO: for each point, find closest centroid index
    labels = []
    for x in X:
        dists = [euclidean_distance(x, c) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels)

def update_centroids(X, labels, k):
    # TODO: new centroid = mean of all points in that cluster
    centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            centroids.append(X[np.random.randint(len(X))])  # handle empty cluster
        else:
            centroids.append(cluster_points.mean(axis=0))
    return np.array(centroids)

def kmeans_scratch(X, k, max_iters=100, tol=1e-4):
    # TODO: Step 1 — random init centroids from data points
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()

    for iteration in range(max_iters):
        # TODO: Step 2 — E-step: assign each point to nearest centroid
        labels = assign_clusters(X, centroids)

        # TODO: Step 3 — M-step: recompute centroids
        new_centroids = update_centroids(X, labels, k)

        # TODO: Step 4 — check convergence
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids
        if shift < tol:
            print(f"  Converged at iteration {iteration+1}")
            break

    # Inertia
    inertia = sum(euclidean_distance(X[j], centroids[labels[j]])**2 for j in range(len(X)))
    return labels, centroids, inertia

labels_scratch, centroids_scratch, inertia_scratch = kmeans_scratch(X_scaled, k=4)
print(f"\n[Scratch] Inertia: {inertia_scratch:.4f}")

# ================================================================
# SECTION 7: VISUALIZE (2D data)
# ================================================================

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels,          cmap='tab10', s=10)
plt.title("sklearn KMeans")
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scratch,  cmap='tab10', s=10)
plt.title("KMeans from scratch")
plt.tight_layout(); plt.savefig("kmeans_compare.png"); plt.close()
print("Comparison plot saved → kmeans_compare.png")

# ================================================================
# QUICK REFERENCE (copy-paste syntax)
# ================================================================
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# km = KMeans(n_clusters=k, n_init=10, random_state=42)
# km.fit(X_scaled)
# labels     = km.labels_
# centroids  = km.cluster_centers_
# inertia    = km.inertia_
# sil        = silhouette_score(X_scaled, labels)
