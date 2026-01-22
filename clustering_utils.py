
import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distance matrix for X (n x d).
    Returns an (n x n) symmetric matrix with zeros on the diagonal.
    
    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Distance matrix of shape (n_samples, n_samples).
    """
    X = np.asarray(X, dtype=np.float64)
    # Using broadcasting for efficiency: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # Note: For numerical stability, sometimes spatial.distance.pdist is better,
    # but this manual implementation is requested/standard in this context.
    sq = np.sum(X**2, axis=1, keepdims=True)  # (n,1)
    D2 = sq + sq.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)  # numerical guard
    return np.sqrt(D2, out=D2)

def make_dataset(seed: int, n_clusters: int = 3, pts_per_cluster: int = 150, spread: float = 1.0) -> np.ndarray:
    """
    Generates 3D synthetic dataset with 'n_clusters' Gaussian blobs.
    Returns an array of shape (n_clusters * pts_per_cluster, 3).
    
    Args:
        seed (int): Random seed.
        n_clusters (int): Number of clusters.
        pts_per_cluster (int): Points per cluster.
        spread (float): Standard deviation of blobs.
        
    Returns:
        np.ndarray: Generated dataset.
    """
    rng = np.random.default_rng(seed)
    cluster_points = []
    for _ in range(n_clusters):
        center = rng.uniform(-10, 10, size=3)
        samples = center + rng.normal(0, spread, size=(pts_per_cluster, 3))
        cluster_points.append(samples)

    X = np.vstack(cluster_points)

    # Ensure at least 500 points as per original notebook logic (if applicable)
    if X.shape[0] < 500:
        extra = rng.uniform(-10, 10, size=(500 - X.shape[0], 3))
        X = np.vstack([X, extra])

    return X

def plot_3d_points(X: np.ndarray, labels: np.ndarray = None, title: str = "3D Scatter") -> None:
    """
    Simple 3D scatter plot. If labels are provided, they are used as colors.
    
    Args:
        X (np.ndarray): Data points (n, 3).
        labels (np.ndarray, optional): Cluster labels.
        title (str): Plot title.
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    if labels is None:
        ax.scatter(X[:,0], X[:,1], X[:,2], s=12)
    else:
        ax.scatter(X[:,0], X[:,1], X[:,2], s=12, c=labels)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(title)
    plt.show()

def detect_outliers(data: np.ndarray, z_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects and removes outliers using Z-score method.

    Args:
        data (np.ndarray): Input data array of shape (n_samples, n_features).
        z_threshold (float): Z-score threshold for outlier detection. Defaults to 3.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Cleaned data (inliers).
            - Indices of outliers.
    """
    if data.size == 0:
        return data, np.array([])
    
    # Calculate Z-scores
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std[std == 0] = 1e-8
    z_scores = np.abs((data - mean) / std)
    
    # Identify outliers (any feature > threshold)
    outliers_mask = (z_scores > z_threshold).any(axis=1)
    inliers = data[~outliers_mask]
    outlier_indices = np.where(outliers_mask)[0]
    
    print(f"Detected {len(outlier_indices)} outliers out of {len(data)} points ({len(outlier_indices)/len(data)*100:.2f}%).")
    return inliers, outlier_indices

def kmeans(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means clustering implementation from scratch.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid convergence.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Cluster labels for each point.
            - Final centroids.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(42)
    # Initialize centroids randomly from data points
    if n_samples >= k:
        centroids = X[rng.choice(n_samples, k, replace=False)]
    else:
        centroids = X # Fallback
        
    labels = np.zeros(n_samples)
    
    for i in range(max_iters):
        # Compute distances to centroids
        # Shape: (n_samples, k)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster: keep old centroid or re-initialize?
                # Keeping old is standard stability behavior.
                new_centroids[j] = centroids[j]
        
        # Check convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"K-Means converged at iteration {i} (shift={shift:.5f})")
            return labels, new_centroids
        centroids = new_centroids
    
    return labels, centroids

def hac(X: np.ndarray, k: int, linkage: str = 'single') -> np.ndarray:
    """
    Hierarchical Agglomerative Clustering implementation using SciPy.

    Args:
        X (np.ndarray): Input data.
        k (int): Number of clusters.
        linkage (str): Linkage criterion ('single', 'complete', 'average', 'ward', 'centroid').

    Returns:
        np.ndarray: Cluster labels.
    """
    # Method mapping: 'centroid' is valid in scipy but check docs.
    # Scipy linkage methods: single, complete, average, weighted, centroid, median, ward.
    try:
        Z = scipy_linkage(X, method=linkage)
    except ValueError as e:
        print(f"HAC Error: {e}. Fallback to 'single'.")
        Z = scipy_linkage(X, method='single')
        
    labels = fcluster(Z, k, criterion='maxclust')
    return labels - 1  # Standardize to 0-based indexing

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes average Silhouette Coefficient from scratch.

    Args:
        X (np.ndarray): Input data.
        labels (np.ndarray): Cluster labels.

    Returns:
        float: Average Silhouette Score.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n = len(X)
    D = pairwise_distances(X)
    
    silhouette_vals = np.zeros(n)
    
    for i in range(n):
        current_cluster = labels[i]
        
        # a(i): Mean distance to same cluster (excluding self)
        # Optimization: pre-calculate indices
        mask_same = (labels == current_cluster)
        # exclude self: mask_same & (indices != i) -- but D[i,i]=0 so sum is same, count is -1
        count_same = np.sum(mask_same)
        if count_same > 1:
            a_i = np.sum(D[i, mask_same]) / (count_same - 1)
        else:
            a_i = 0
        
        # b(i): Min mean distance to other clusters
        b_i = np.inf
        for label in unique_labels:
            if label == current_cluster:
                continue
            mask_other = (labels == label)
            if np.sum(mask_other) > 0:
                mean_dist = np.mean(D[i, mask_other])
                if mean_dist < b_i:
                    b_i = mean_dist
        
        if b_i == np.inf:
            b_i = 0 
            
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        
    return np.mean(silhouette_vals)
