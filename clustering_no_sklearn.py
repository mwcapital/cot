"""
Alternative Participant Behavior Clusters implementation without sklearn
Uses numpy and scipy only for clustering
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def standardize_features(features):
    """Standardize features to zero mean and unit variance"""
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    # Avoid division by zero
    stds[stds == 0] = 1
    return (features - means) / stds

def simple_kmeans(data, n_clusters, max_iter=100, random_state=42):
    """Simple K-means implementation using numpy"""
    np.random.seed(random_state)
    n_samples, n_features = data.shape
    
    # Initialize centroids randomly
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = data[indices]
    
    labels = np.zeros(n_samples, dtype=int)
    
    for iteration in range(max_iter):
        # Assign points to nearest centroid
        distances = cdist(data, centroids, metric='euclidean')
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
            
        labels = new_labels
        
        # Update centroids
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                centroids[i] = np.mean(data[mask], axis=0)
    
    return labels, centroids

def alternative_pca(data, n_components=2):
    """Simple PCA implementation using numpy"""
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data
    components = eigenvectors[:, :n_components]
    projected = data_centered @ components
    
    # Calculate explained variance
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return projected, components, explained_variance_ratio

# Demo usage
if __name__ == "__main__":
    print("CLUSTERING WITHOUT SKLEARN - DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create 3 clusters with different characteristics
    cluster1 = np.random.normal(loc=[0.5, 1, 2, 30, 60], scale=0.2, size=(30, n_features))
    cluster2 = np.random.normal(loc=[0.2, -1, 8, 20, 45], scale=0.3, size=(40, n_features))
    cluster3 = np.random.normal(loc=[2.0, 0, 1.5, 80, 70], scale=0.25, size=(30, n_features))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"\nCreated {n_samples} samples with {n_features} features")
    
    # Standardize
    print("\n1. Standardizing features...")
    data_scaled = standardize_features(data)
    
    # Cluster
    print("\n2. Performing clustering...")
    labels, centroids = simple_kmeans(data_scaled, n_clusters=3)
    
    # Analyze clusters
    print("\n3. Cluster Analysis:")
    for i in range(3):
        mask = labels == i
        cluster_data = data[mask]
        print(f"\nCluster {i} ({np.sum(mask)} members):")
        print(f"  Feature means: {np.mean(cluster_data, axis=0)}")
    
    # PCA for visualization
    print("\n4. Dimensionality reduction (PCA)...")
    projected, components, explained_var = alternative_pca(data_scaled)
    print(f"  Explained variance: {explained_var * 100}")
    
    # Alternative: Hierarchical clustering
    print("\n5. Alternative: Hierarchical clustering...")
    linkage_matrix = linkage(data_scaled, method='ward')
    hierarchical_labels = fcluster(linkage_matrix, 3, criterion='maxclust')
    
    print("\nComparison:")
    print(f"  K-means labels: {labels[:10]}...")
    print(f"  Hierarchical labels: {hierarchical_labels[:10]}...")
    
    print("\n✅ Successfully demonstrated clustering without sklearn!")