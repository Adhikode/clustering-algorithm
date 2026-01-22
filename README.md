# Clustering Algorithms Implementation

Implementation of K-means and Hierarchical Agglomerative Clustering (HAC) algorithms from scratch for CS634 Data Mining course at NJIT.

## Author
**Niraj Adhikari**  
PhD Student, New Jersey Institute of Technology  
Fall 2025 - CS634 Data Mining Final Project

## Overview

This project implements two fundamental clustering algorithms:
- **K-means Clustering**: Partitioning method that divides data into k distinct clusters
- **Hierarchical Agglomerative Clustering (HAC)**: Bottom-up approach with three linkage methods (single, complete, average)

Additionally implements:
- Outlier detection using Z-score method
- Silhouette score for clustering quality assessment
- Validation against scikit-learn implementations

## Repository Structure

```
Clustering/
├── FinalProject_Clustering_v1.ipynb  # Main Jupyter notebook
├── data/                              # Generated datasets
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── dataset3.csv
├── adhikari_niraj_finaltermproj.pdf  # Project report
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clustering-implementation.git
cd clustering-implementation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

```bash
jupyter notebook FinalProject_Clustering_v1.ipynb
```

###Example: K-means Clustering

```python
import numpy as np
from typing import Tuple

# Load your implementation
# ... (paste relevant functions from notebook)

# Generate or load data
X = np.random.randn(100, 3)

# Run K-means
labels, centroids = kmeans(X, k=3, max_iter=100, tol=1e-5)

# Evaluate clustering quality
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")
```

### Example: Hierarchical Agglomerative Clustering

```python
# Run HAC with average linkage
labels = hac(X, k=3, linkage="average")

# Visualize results (for 3D data)
plot_3d_points(X, labels=labels, title="HAC Clustering Result")
```

## Features

### 1. K-means Clustering
- Efficient vectorized implementation using NumPy
- Convergence monitoring with tolerance threshold
- Iteration counter with maximum iteration limit
- Returns cluster labels and final centroids

### 2. Hierarchical Agglomerative Clustering
- Three linkage methods: single, complete, average
- Efficient distance computation
- Builds complete dendrogram internally
- Cuts tree at specified number of clusters

### 3. Outlier Detection
- Z-score based method
- Configurable threshold (default: 2.5)
- Returns cleaned dataset and outlier indices

### 4. Silhouette Score
- Assessment of clustering quality
- Computed from scratch (no sklearn dependency for this metric)
- Range: [-1, 1], higher is better

## Datasets

Three synthetic 3D datasets with varying cluster spreads:
- **Dataset 1**: Moderate spread (σ=1.8) - 450 clustered + 50 random points
- **Dataset 2**: High spread (σ=3.1) - 450 clustered + 50 random points  
- **Dataset 3**: Low spread (σ=1.0) - 450 clustered + 50 random points

All datasets have 3 Gaussian clusters with 150 points each.

## Results Summary

### K-means Performance (Dataset 1, after outlier removal)
```
Our K-Means silhouette:      0.5734
sklearn KMeans silhouette:   0.5731
```

### HAC Performance (Dataset 1, after outlier removal)
```
Linkage = single   | ours = 0.1265 | sklearn = 0.1222
Linkage = complete | ours = 0.3695 | sklearn = 0.3690
Linkage = average  | ours = 0.5701 | sklearn = 0.5698
```

Results show excellent agreement with sklearn implementations, validating correctness of our algorithms.

## Key Functions

- `pairwise_distances(X)` - Compute Euclidean distance matrix
- `kmeans(X, k, max_iter, tol)` - K-means clustering
- `hac(X, k, linkage)` - Hierarchical agglomerative clustering
- `silhouette_score(X, labels)` - Clustering quality metric
- `detect_outliers(X, z_threshold)` - Z-score based outlier detection
- `plot_3d_points(X, labels, title)` - 3D visualization

## Dependencies

- NumPy ≥ 1.24.0 - Numerical computations
- Matplotlib ≥ 3.7.0 - Visualization
- SciPy ≥ 1.10.0 - Hierarchical clustering utilities
- scikit-learn ≥ 1.3.0 - Validation only (not used in core implementation)

## Validation

All implementations are validated against scikit-learn:
- K-means validated using `sklearn.cluster.KMeans`
- HAC validated using `sklearn.cluster.AgglomerativeClustering`
- Silhouette scores validated using `sklearn.metrics.silhouette_score`

Validation results show differences of < 0.5%, confirming implementation correctness.

## Future Improvements

- Add more clustering algorithms (DBSCAN, GMM)
- Implement elbow method for optimal k selection
- Add dendrogram visualization for HAC
- Modularize code into separate Python files
- Add comprehensive unit tests

## License

This project is created for educational purposes as part of CS634 coursework.

## Acknowledgments

- Course: CS634 Data Mining
- Institution: New Jersey Institute of Technology
- Instructor: Prof. Zhenduo Wang
- Semester: Fall 2025

## Contact

For questions or collaboration:
- Email: na765@njit.edu
- GitHub: (https://github.com/Adhikode)

---

**Note**: This implementation is for educational purposes and may not be optimized for production use.
