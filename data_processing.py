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
    
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X, y), _ = fashion_mnist.load_data()

    rng = np.random.default_rng(42)
    n_samples = 1000

    chosen_indices = rng.choice(len(X), size=n_samples, replace=False)

    X_sampled = X[chosen_indices]
    y_sampled = y[chosen_indices]
    return X_sampled, y_sampled
def load_synthetic_circle_data(): #just downloaded it lol
    df = np.loadtxt("data/circles.txt", delimiter=",", skiprows=1)
    X = df[:, :2]
    y = df[:, -1]
    return X, y

def image_preprocessing(X):
    # normalize the pixel values to the range [0, 1]
    samples, height, width = X.shape[0], X.shape[1], X.shape[2] #flattening
    X_flattened = X.reshape(samples, height * width)
    X_normalized = X_flattened.astype('float32') / 255.0 #normilizing to [0, 1]
    return X_normalized
    
def synthetic_data_preprocessing(X):
    X_normalized = X.astype('float32') / np.max(X) #normilizing to [0, 1]
    return X_normalized

if __name__ == "__main__":
    X_mnist, y_mnist = load_fashion_mnist()
    print("Loaded X shape:", X_mnist.shape)
    print(X_mnist[:5])
    print(y_mnist[:5])
    X_mnist_processed = image_preprocessing(X_mnist)
    print("Normalised X:", X_mnist_processed[:5])
    print("y:", y_mnist[:5])

    X_synth, y_synth = load_synthetic_circle_data()
    print("Loaded X shape:", X_synth.shape)
    print(X_synth[:5])
    print(y_synth[:5])
    X_synth_processed = synthetic_data_preprocessing(X_synth)
    print("Normalised X:", X_synth_processed[:5])
    print("y:", y_synth[:5])