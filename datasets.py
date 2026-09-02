import os, re
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import fashion_mnist #used only for datset loading

CLASS_NAMES = {
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

def label_encoding(y: np.ndarray) -> np.ndarray:
    return np.array([CLASS_NAMES[label] for label in y])


def load_fashion_mnist():
    
    (X, y), _ = fashion_mnist.load_data()

    rng = np.random.default_rng(42)
    n_samples = 1000

    chosen_indices = rng.choice(len(X), size=n_samples, replace=False)

  
    X_sampled = X[chosen_indices]
    return X_sampled 

def scale_pixels(X):
    # normalize the pixel values to the range [0, 1]
    # normalize the pixel values to the range [0, 1]
    samples, height, width = X.shape[0], X.shape[1], X.shape[2] #flattening
    X_flattened = X.reshape(samples, height * width)
    X_normalized = X_flattened.astype('float32') / 255.0 #normilizing to [0, 1]
    return X_normalized

def random_projection(X, n_components):
    # to reduce dimensionality 
    """
    Johnson-Lindenstrauss lemma:
    If you project points onto enough random directions, 
    the distances between them are approximately preserved.
    """
    rng = np.random.default_rng(42)
    random_matrix = rng.normal(size=(X.shape[1], n_components)) / np.sqrt(n_components) #read it on here https://mdp-toolkit.github.io/node_list.html
    X_projected = np.dot(X, random_matrix)
    return X_projected

if __name__ == "__main__":
    X_loaded = load_fashion_mnist()
    X_norm = scale_pixels(X_loaded)
    X_proj = random_projection(X_norm, n_components=50)
    print("Loaded:", X_loaded.shape, "Normalized:", X_norm.shape, "Projected:", X_proj.shape)
    print("First 5 samples of projected data:\n", X_proj[:5])
