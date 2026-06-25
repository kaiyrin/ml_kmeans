import numpy as np

def first_centroid(X):
    """Select the first centroid randomly from the dataset."""
    total_images = len(X)
    random_index = np.random.choice(total_images)
    return X[random_index]



