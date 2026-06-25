import os
import numpy as np
 
"""
Logic behind that:
1. Select k no. of clusters in my case 10 for mnsit need silhuete for optimal k for synthetic data
2. Randonly select k=10 data points as initial centroids. *initial centroids*
3. Measure the distance between each data point and the centroids. 
4. Assign each data point to the nearest centroid. *assigning clusters*
5. Calculate mean of each cluster *mean cluster*
6. Update the centroids. *updating centroids*
7. New starting points as new centroids assigned to each cluster. *new centroids*


So, distinct methods to call:
1. initialize_centroids(X, k)
2. Euclidean_distance(X, centroids)
2. assign_clusters(distances) 
3. mean_image_of_cluster(X, labels, k) OR center_of_cluster(X, labels, k) => update centroids

[iterate until convergence]
"""

def random_init_centroids(X, k): #i have seen other implementation with min and max darkness points found it pretty long also since my datset already smothly distributed lets hope itll cover each labels
    total_images = len(X)
    random_indices = np.random.choice(total_images, size=k, replace=False)
    centroids = X[random_indices]
    return centroids

def euclidean_distance(X, centroids, k): #code from here [LINK:https://github.com/tugot17/K-Means-Algorithm-From-Scratch/blob/master/K-means.ipynb]to compare with mine
    """# 1. Subtract the centroid from all images: (X - centroids[i])
        # 2. Square the differences: ** 2
        # 3. Sum across the 784 pixels (axis=1): np.sum(...)
        # 4. Take the square root: np.sqrt(...)"""
    total_images = len(X)
    distances = np.zeros((total_images, k))
    for i in range(k):
        squared_diff = (X - centroids[i]) ** 2
        sum_squared_diff = [sum(squared_diff[j]) for j in range(total_images)]
        distances[:, i] = np.sqrt(sum_squared_diff)
    return distances

def assign_clusters(distances):
    labels = [list(row).index(min(row)) for row in distances] #instaead of labels = np.argmin(distances, axis=1)
    return labels

def mean_image_of_cluster(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[np.array(labels) == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = np.zeros(X.shape[1])  # hndle empty clusters
        new_centroids.append(new_centroid)
    centroids = np.array(new_centroids)
    return centroids    

if __name__ == "__main__":
    random_init_centroids(np.random.rand(100, 2), 10)
    euclidean_distance(np.random.rand(100, 2), np.random.rand(10, 2), 10)
    assign_clusters(np.random.rand(100, 10))
    mean_image_of_cluster(np.random.rand(100, 2), [0]*50 + [1]*50, 2)
    
