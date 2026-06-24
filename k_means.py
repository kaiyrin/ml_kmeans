import os
import numpy as np
 
"""
Logic behind that:
1. Select k no. of clusters in my case 10
2. Randonly select k=10 data points as initial centroids. *initial centroids*
3. Measure the distance between each data point and the centroids. 
4. Assign each data point to the nearest centroid. *assigning clusters*
5. Calculate mean of each cluster *mean cluster*
6. Update the centroids. *updating centroids*
7. New starting points as new centroids assigned to each cluster. *new centroids*


So, distinct methods to call:
1. initialize_centroids(X, k)
2. Euclidean_distance(X, centroids)
2. assign_clusters(X, centroids) 
3. mean_cluster(X, labels, k) OR center_of_cluster(X, labels, k)
4. update_centroids(X, labels, k)
[iterate until convergence]
"""

def randomly_initialize_centroids(X, k): #i have seen other implementation with min and max darkness points found it pretty long also since my datset already smothly distributed lets hope itll cover each labels
    """Randomly select k data points as initial centroids."""
    total_images = len(X)
    random_indices = np.random.choice(total_images, size=k, replace=False)
    centroids = X[random_indices] #or    centroids = [X[i] for i in random_indices] but i know i have a numpy array so
    return centroids

def euclidean_distance(X, centroids):
    """Calculate the Euclidean distance between each data point and the centroids."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances

def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid."""
    distances = euclidean_distance(X, centroids)
    labels = np.argmin(distances, axis=1)
    return labels

def mean_cluster(X, labels, k):
    """Calculate the mean of each cluster."""
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids    

if __name__ == "__main__":
    pass