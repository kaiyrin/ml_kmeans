import numpy as np
from collections import defaultdict

from datasets import (load_fashion_mnist, 
                      scale_pixels, 
                      random_projection,
                      label_encoding, 
                      new_label_mapping,
                      create_synthetic_dataset,
                      FASHION_MNIST_NAMES,
                      SYNTH_NAMES)
from k_means import k_means_run, compute_inertia


class KMeansExperiment:
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(dict))
        self.data = {} 
        self.summary = []

    def run_experiment(self, method, dataset_type, k, max_iter, seed): # Changed 'seeds' to 'seed'
        if dataset_type == "fashion_mnist":
            X_original_image, y_true = load_fashion_mnist()
            X = scale_pixels(X_original_image)
            # applying random projection using the single seed
            X, random_matrix = random_projection(X, n_components=50, seed=seed)
        elif dataset_type == "synthetic":
            X, y_true = create_synthetic_dataset(n_samples = 1000, n_features = 50, n_classes = 10)
        else:
            pass

        #running k-means either with kmeans++ or kmeans
        centroids, new_y = k_means_run(X, k, max_iter, seed, method = "kmeans_plusplus" if method == "kmeans_plusplus" else "kmeans_simple")
        new_y = np.array(new_y)
        mapped_new_y = new_label_mapping(y_true, new_y, k)
        accuracy = np.sum(mapped_new_y == y_true) / len(y_true)

        names = SYNTH_NAMES if dataset_type == "synthetic" else FASHION_MNIST_NAMES
        y_new_names  = label_encoding(mapped_new_y, names)
        y_true_names = label_encoding(y_true, names)
        #inertia 
        inertia = compute_inertia(X, centroids, k)

        self.results[method][dataset_type][seed] = {
            "x_original": X_original_image if dataset_type == "fashion_mnist" else X,
            "x_projected": X,
            "y": y_true,
            "y_clustered_id": new_y,
            "y_label_name": y_true_names,   
            "y_mapped": mapped_new_y,      
            "y_new_label_name": y_new_names,
            "accuracy": accuracy,
            "centroids": centroids,
            "inertia": inertia
        }
        
        self.summary.append({
            "method": method,
            "dataset": dataset_type,
            "seed": seed,
            "accuracy": accuracy,
            "inertia": inertia
        })
        
        return self


if __name__ == "__main__":
    experiment = KMeansExperiment()
    """
    experiment.run_experiment(method="kmeans_simple", dataset_type="fashion_mnist", k=10, max_iter=100, seed=42)
    print(experiment.summary)
    """
    experiment.run_experiment(method="kmeans_plusplus", dataset_type="synthetic", k=10, max_iter=100, seed=42)
    print(experiment.summary)