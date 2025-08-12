"""
A1. Evaluate the intraclass spread and interclass distances between the classes in your dataset.
If your data deals with multiple classes, you can take any two classes.

Steps:
    • Calculate the mean for each class (class centroid)
      - Use numpy.mean(..., axis=0) to get the average feature vector for a class.
    • Calculate spread (standard deviation) for each class
      - Use numpy.std(..., axis=0) to compute the variation within each class.
    • Calculate the distance between mean vectors (centroids) of classes
      - Use numpy.linalg.norm(centroid1 - centroid2) for Euclidean distance.

This script:
    - Loads an image dataset (MiniImageNet-style format) with two classes.
    - Converts images to flattened vectors.
    - Computes class centroids, average spread, and centroid distance.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from numpy.linalg import norm


class ClassSpreadEvaluator:
    """Class to evaluate intra-class spread and inter-class distances."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        """
        Initialize dataset path and transformation pipeline.

        Args:
            dataset_path (str): Path to the dataset folder (ImageFolder format).
            image_size (tuple): Target size to resize images (default 84x84).
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.image_dataset = None
        self.class_features = {}

    def load_dataset(self):
        """Load and transform dataset into flattened feature vectors."""
        transform = transforms.Compose([
            transforms.Resize(self.image_size),     # Resize all images to uniform size
            transforms.ToTensor(),                  # Convert image to PyTorch tensor
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image to a vector
        ])

        # Load dataset from folder structure (class_name/subfolder_name)
        self.image_dataset = datasets.ImageFolder(self.dataset_path, transform=transform)
        print(f"✅ Dataset loaded with {len(self.image_dataset)} samples and {len(self.image_dataset.classes)} classes.")

    def separate_by_class(self):
        """Separate image features into class-specific NumPy arrays."""
        # Initialize dictionary for each class index
        self.class_features = {class_idx: [] for class_idx in range(len(self.image_dataset.classes))}

        # Iterate through dataset and store features
        for image, label in self.image_dataset:
            self.class_features[label].append(image.numpy())

        # Convert lists to NumPy arrays
        for class_idx in self.class_features:
            self.class_features[class_idx] = np.array(self.class_features[class_idx])
            print(f"📂 Class {class_idx} -> Shape: {self.class_features[class_idx].shape}")

    def compute_centroid(self, class_idx: int):
        """Compute the mean vector (centroid) for a given class."""
        return np.mean(self.class_features[class_idx], axis=0)

    def compute_spread(self, class_idx: int):
        """Compute the average spread (mean of standard deviations) for a given class."""
        std_vector = np.std(self.class_features[class_idx], axis=0)
        return np.mean(std_vector)

    def compute_centroid_distance(self, class_idx1: int, class_idx2: int):
        """Compute Euclidean distance between centroids of two classes."""
        centroid1 = self.compute_centroid(class_idx1)
        centroid2 = self.compute_centroid(class_idx2)
        return norm(centroid1 - centroid2)

    def evaluate(self, class_idx1=0, class_idx2=1):
        """Perform full evaluation for two given classes."""
        # Compute centroids
        centroid_0 = self.compute_centroid(class_idx1)
        centroid_1 = self.compute_centroid(class_idx2)

        # Compute spreads
        spread_0 = self.compute_spread(class_idx1)
        spread_1 = self.compute_spread(class_idx2)

        # Compute centroid distance
        distance = self.compute_centroid_distance(class_idx1, class_idx2)

        # Display results
        print("\n📊 Evaluation Results:")
        print(f"Class {class_idx1} Centroid Shape: {centroid_0.shape}")
        print(f"Class {class_idx2} Centroid Shape: {centroid_1.shape}")
        print(f"Class {class_idx1} Average Spread (Std Dev): {spread_0}")
        print(f"Class {class_idx2} Average Spread (Std Dev): {spread_1}")
        print(f"Distance Between Class Centroids: {distance}")


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset" 
    evaluator = ClassSpreadEvaluator(dataset_path)
    evaluator.load_dataset()
    evaluator.separate_by_class()
    evaluator.evaluate(class_idx1=0, class_idx2=1)
