import os, re
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import fashion_mnist #used only for datset loading 

FASHION_MNIST_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}
SYNTH_NAMES = {
    0: "Alpha",
    1: "Beta",
    2: "Gamma",
    3: "Delta",
    4: "Epsilon",
    5: "Zeta",
    6: "Eta",
    7: "Theta",
    8: "Iota",
    9: "Kappa",
}

def label_encoding(y: np.ndarray, names=None) -> np.ndarray:
    if names is None:
        names = FASHION_MNIST_NAMES
    return np.array([names[label] for label in y])
def new_label_mapping(y, new_y, k) -> np.ndarray:
    mapped_new_y = np.zeros_like(new_y)
    for i in range(k):
        cluster_indeces = np.where(new_y == i)[0]
        if len(cluster_indeces) > 0:
            true_labeling = y[cluster_indeces]
            most_common_label = np.bincount(true_labeling).argmax()
            mapped_new_y[cluster_indeces] = most_common_label
    return mapped_new_y

def load_fashion_mnist():
    
    (X, y), _ = fashion_mnist.load_data()

    rng = np.random.default_rng(42)
    n_samples = 1000

    chosen_indices = rng.choice(len(X), size=n_samples, replace=False)

  
    X_sampled = X[chosen_indices]
    y_sampled = y[chosen_indices]
    return X_sampled, y_sampled

def scale_pixels(X):
    samples, height, width = X.shape #flattening
    X_flattened = X.reshape(samples, height * width)
    X_normalized = X_flattened.astype('float32') / 255.0 #normilizing to [0, 1]
    return X_normalized

def random_projection(X, n_components,seed):
    # to reduce dimensionality 
    """
    Johnson-Lindenstrauss lemma:
    If you project points onto enough random directions, 
    the distances between them are approximately preserved.
    """
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(X.shape[1], n_components)) / np.sqrt(n_components) #read it on here https://mdp-toolkit.github.io/node_list.html
    X_projected = np.dot(X, random_matrix)
    return X_projected, random_matrix

def create_synthetic_dataset(n_samples, n_features, n_classes):
    rng = np.random.default_rng(42)
    random_centers = rng.uniform(low=-5, high=5, size=(n_classes, n_features)) #centers far apart not much overlap need to do less later if too easy for k-mean idk which to put for 50 dimensions
    counts = np.full(random_centers.shape[0], n_samples // n_classes) #`counts = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    X_partial, y_partial = [], []
    for i, center in enumerate(random_centers):
        cluster_samples = rng.normal(loc=center, scale=3.0, size=(counts[i], n_features)) #std to sprad points from each other
        X_partial.append(cluster_samples)
        y_partial.append(np.full(counts[i], i))
    X = np.vstack(X_partial) #1000, 50)
    y = np.concatenate(y_partial)
    # shuffle the dataset
    indices = rng.permutation(len(X))
    X_synth = X[indices]
    y_synth = y[indices]
    return X_synth, y_synth 





if __name__ == "__main__":
    X_loaded, y_mnist = load_fashion_mnist()
    X_norm = scale_pixels(X_loaded)
    X_minst = random_projection(X_norm, n_components=50, seed=42)
    print("Loaded:", X_loaded.shape, "Normalized:", X_norm.shape, "Projected:", X_minst.shape)
    print("First 5 samples of projected MNIST data:\n", X_minst[:5])
    print("First 5 labels of projected MNIST data:\n", y_mnist[:5])
    X_synth, y_synth = create_synthetic_dataset(
        n_samples=1000,
        n_features=50,
        n_classes=10,
    )
    print("Synthetic dataset shape:", X_synth.shape, "Labels shape:", y_synth.shape)
    print("First 5 samples of synthetic data:\n", X_synth[:5])
    print("First 5 labels of synthetic data:\n", y_synth[:5])