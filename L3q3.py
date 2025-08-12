"""
A3. Take any two feature vectors from your dataset.
    - Calculate the Minkowski distance for r from 1 to 10.
    - Make a plot of the distance and observe the nature of the graph.
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class MinkowskiDistanceAnalyzer:
    """Class to compute Minkowski distances between two feature vectors and split dataset."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.dataset = None
        self.features = None
        self.labels = None

    def load_dataset(self):
        """Load dataset and flatten images into feature vectors."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))
        ])
        self.dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)

        # Extract features and labels
        self.features = np.array([img.numpy() for img, _ in self.dataset])
        self.labels = np.array([label for _, label in self.dataset])

        print(f"✅ Loaded {len(self.dataset)} images across {len(set(self.labels))} classes.")

    def compute_minkowski_distances(self, vector1_idx=0, vector2_idx=1, r_max=10):
        """
        Compute Minkowski distances between two selected feature vectors.

        Args:
            vector1_idx (int): Index of first vector.
            vector2_idx (int): Index of second vector.
            r_max (int): Maximum r value (default 10).
        """
        vec1 = self.features[vector1_idx]
        vec2 = self.features[vector2_idx]

        r_values = list(range(1, r_max + 1))
        distances = [
            np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r) for r in r_values
        ]

        # Plot distances
        plt.plot(r_values, distances, marker='o')
        plt.title("Minkowski Distance vs r")
        plt.xlabel("r (Minkowski Power Parameter)")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.show()

        # Print results
        for r, dist in zip(r_values, distances):
            print(f"r = {r}: Distance = {dist:.4f}")

    def split_dataset(self, test_size=0.3):
        """
        Split dataset into train and test sets (only two classes).

        Args:
            test_size (float): Proportion for test split (default 0.3).
        """
        # If dataset has more than 2 classes, select only first 2
        unique_classes = sorted(set(self.labels))
        if len(unique_classes) > 2:
            print("⚠ Multi-class dataset detected. Selecting only first two classes.")
            mask = np.isin(self.labels, unique_classes[:2])
            X = self.features[mask]
            y = self.labels[mask]
        else:
            X = self.features
            y = self.labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"📊 Train set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"  

    analyzer = MinkowskiDistanceAnalyzer(dataset_path)
    analyzer.load_dataset()
    
    # A3: Minkowski distance computation
    analyzer.compute_minkowski_distances(vector1_idx=0, vector2_idx=1, r_max=10)

    # A4: Train-test split
    X_train, X_test, y_train, y_test = analyzer.split_dataset(test_size=0.3)
