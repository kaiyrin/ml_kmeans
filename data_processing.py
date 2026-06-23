import os, re
import numpy as np
from tensorflow.keras.datasets import fashion_mnist #used only for datset loading


def load_fashion_mnist():
    for cls in classes:
        idx = np.where(y == cls)[0]           
        chosen = rng.choice(idx, size=100)    
        X_parts.append(X[chosen])
        y_parts.append(y[chosen])            

    return X, y
def data_prepreocessing(X):
    # Normalize the pixel values to the range [0, 1]
    samples, height, width = X.shape[0], X.shape[1], X.shape[2]
    X_flattened = X.reshape(samples, height * width)
    X_normalized = X_flattened.astype('float32') / 255.0
    
    
    return X_normalized


if __name__ == "__main__":
    X, y = load_fashion_mnist()
    X = data_prepreocessing(X)
