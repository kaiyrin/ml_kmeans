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

"""
K means plus plus initialization:
1. Randomly select the first centroid from the data points here i called the random_init_centroids function but with k=1 for only first step
2. For new centeroid c_i from X data points using probaility distribution weight P(x_n) = D(x_n)/sum(D(x_n)) where D(x_n) is the squared distance to the nearest centroid, and then select the next centroid with probability proportional to D². 
   This ensures that points that are farther away from existing centroids have a higher chance of being selected as the next centroid.
   D(x_n) = min(||x_n - c_1||^2, ||x_n - c_2||^2, ..., ||x_n - c_i-1||^2)
3. Repeat step 2 until k centroids are selected.
"""
#K-means regular initialization function
def random_init_centroids(X, k): #i have seen other implementation with min and max darkness points found it pretty long also since my datset already smothly distributed lets hope itll cover each labels
    total_images = len(X)
    random_indices = np.random.choice(total_images, size=k, replace=False)
    centroids = X[random_indices]
    return centroids
#K-means plus plus initialization function
def plusplus_init_centroids(X, k): # D² — the squared distance to the nearest centroid, and then select the next centroid with probability proportional to D². This ensures that points that are farther away from existing centroids have a higher chance of being selected as the next centroid.
    total_images = len(X)
    centroids = []
    # randomly select the first centroid from the previous funtion 
    centroids = list(random_init_centroids(X, 1))
    
    for _ in range(1, k):
        newest_centroid = centroids[-1]
        # calculate the squared distances to the nearest centroid
        xc_squared = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
        d_x = np.min(xc_squared, axis=0)  # minimum distance to the nearest centroid
        probabilities = d_x / d_x.sum()
        """ To check the previous erro occured with Exception has occurred: ValueError 'a' and 'p' must have same size 
        print(len(probabilities))
        print(len(d_x))
        print(len(X))
        """
        next_centroid_index = np.random.choice(total_images, p=probabilities)
        centroids.append(X[next_centroid_index])
    
    return np.array(centroids)



def euclidean_distance(X, centroids, k): #code from here [LINK:https://github.com/tugot17/K-Means-Algorithm-From-Scratch/blob/master/K-means.ipynb]to compare with mine
    """# 1. Subtract the centroid from all images: (X - centroids[i])
        # 2. Square the differences: ** 2
        # 3. Sum across the 784 pixels (axis=1): np.sum(...)
        # 4. Take the square root: np.sqrt(...)"""
    total_images = len(X)
    distances = np.zeros((total_images, k))
    for i in range(k):
        squared_diff = (X - centroids[i]) ** 2
        sum_squared_diff = np.sum(squared_diff, axis=1)
        distances[:, i] = np.sqrt(sum_squared_diff) #here used to be 0
    return distances

def assign_clusters(distances):
    labels = [list(row).index(min(row)) for row in distances] #cpuld have been labels = np.argmin(distances, axis=1) for memory optimization but i dint get it
    return labels

def mean_image_of_cluster(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[np.array(labels) == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            #new_centroid = np.zeros(X.shape[1]) #first i handled empty cluster by setting it to zero 
            #pick random exisitn gpoint to get out of empty cluster situation
            new_centroid = X[np.random.choice(len(X))]  # https://datascience.stackexchange.com/questions/9898/what-to-do-with-stale-centroids-in-k-means
        new_centroids.append(new_centroid)
    centroids = np.array(new_centroids)
    return centroids    
def k_means_run(X, k, max_iter, seed, method):
    np.random.seed(seed)
    if method == "kmeans_plusplus":
        centroids = plusplus_init_centroids(X, k)
    else:
        centroids = random_init_centroids(X, k)
    
    for iteration in range(max_iter):
        distances = euclidean_distance(X, centroids, k)
        new_y = assign_clusters(distances)
        new_centroids = mean_image_of_cluster(X, new_y, k)
    
        #checking for convergence
        centroid_shift = np.sum((new_centroids - centroids) ** 2)
        if centroid_shift < 1e-6:
            print(f"Converged after {iteration} iterations.")
            break
        centroids = new_centroids
    return centroids, new_y

def compute_inertia(X, centroids, k):
    distances = euclidean_distance(X, centroids, k)   
    nearest = np.min(distances, axis=1)               # distance to assigned centroid
    return np.sum(nearest ** 2)                        # square to undo the sqrt


if __name__ == "__main__":
    """
    random_init_centroids(np.random.rand(100, 2), 10)
    euclidean_distance(np.random.rand(100, 2), np.random.rand(10, 2), 10)
    assign_clusters(np.random.rand(100, 10))
    mean_image_of_cluster(np.random.rand(100, 2), [0]*50 + [1]*50, 2)
    print((np.random.rand(100, 2), 10, 100))
    print(compute_inertia(np.random.rand(100, 2), np.random.rand(10, 2), 10))
    """
    print("K-means++ initialization:")
    centroids = plusplus_init_centroids(np.random.rand(100, 2), 10)
    print(centroids)    
    print("K-means run:")
    centroids, labels = k_means_run(np.random.rand(100, 2), 10, 100, 42, "kmeans_plusplus")
    print(centroids)
    print(labels)
    
    
