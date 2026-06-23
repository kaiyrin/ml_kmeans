import os, re
import numpy as np
from tensorflow.keras.datasets import fashion_mnist #used only for datset loading


def load_fashion_mnist():
    
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X, y), _ = fashion_mnist.load_data()

    rng = np.random.default_rng(42)
    n_samples = 1000

    chosen_indices = rng.choice(len(X), size=n_samples, replace=False)

    X_sampled = X[chosen_indices]
    y_sampled = y[chosen_indices]
    return X_sampled, y_sampled

def image_preprocessing(X):
    # normalize the pixel values to the range [0, 1]
    samples, height, width = X.shape[0], X.shape[1], X.shape[2]
    X_flattened = X.reshape(samples, height * width)
    X_normalized = X_flattened.astype('float32') / 255.0
    
    
    return X_normalized


if __name__ == "__main__":
    X, y = load_fashion_mnist()
    print(X[:5])
    print(y[:5])
    X_processed = image_preprocessing(X)
    print("Normalised X:", X_processed[:5])
    print("y:", y[:5])
