"""
A8. Make k = 1 to implement NN classifier and compare the results with kNN (k = 3).
    - Vary k from 1 to 11
    - Train and evaluate a k-NN classifier for each k
    - Record and plot accuracy vs k
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNKComparison:
    """Class to train and compare k-NN performance for varying k values."""

    def __init__(self, dataset_path, image_size=(84, 84)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k_values = list(range(1, 12))  # k from 1 to 11
        self.k_accuracies = []

    def load_and_split_data(self):
        """Load dataset, preprocess images, and split into train/test sets."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))  # Flatten to 1D vector
        ])

        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"✅ Data Loaded: {len(self.X_train)} train, {len(self.X_test)} test")

    def evaluate_for_k_values(self):
        """Train k-NN for different k values and record accuracy."""
        for k in self.k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)

            predictions = knn.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)

            self.k_accuracies.append(accuracy)
            print(f"k = {k}, Accuracy = {accuracy:.4f}")

    def plot_accuracy_vs_k(self):
        """Plot accuracy scores for different k values."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_values, self.k_accuracies, marker='o', linestyle='-')
        plt.title('k-NN Accuracy vs k')
        plt.xlabel('k (Number of Neighbors)')
        plt.ylabel('Accuracy')
        plt.xticks(self.k_values)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------- Main Execution ----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"  

    knn_comparison = KNNKComparison(dataset_path)
    knn_comparison.load_and_split_data()
    knn_comparison.evaluate_for_k_values()
    knn_comparison.plot_accuracy_vs_k()
