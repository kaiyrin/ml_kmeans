import os, re
import pandas as pd

from datasets import load_fashion_mnist, scale_pixels, random_projection
      



def main():
    X_loaded = load_fashion_mnist()
    X_norm = scale_pixels(X_loaded)
    X_proj = random_projection(X_norm, n_components=50)
    print( X_proj[:5])



if __name__ == "__main__":
    main()