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

from torchvision import datasets, transforms   # Import datasets and transforms from torchvision
from torch.utils.data import DataLoader        # Import DataLoader to load data in batches
import numpy as np                             # Import NumPy for numerical operations
from numpy.linalg import norm                  # Import norm function to calculate  Euclidean distance.



class ClassSpreadEvaluator:
    """Class to evaluate intra-class spread and inter-class distances."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        """
        Initialize dataset path and transformation pipeline.

        Args:
            dataset_path (str): Path to the dataset folder (ImageFolder format).
            image_size (tuple): Target size to resize images (default 84x84).
        """
        self.dataset_path = dataset_path #where your images are stored.
        self.image_size = image_size #size to which all images will be resized.
        self.image_dataset = None # Placeholder for loaded dataset
        self.class_features = {} # Dictionary will store NumPy arrays of flattened images for each class.

    def load_dataset(self):
        """Load and transform dataset into flattened feature vectors."""
        transform = transforms.Compose([
            transforms.Resize(self.image_size),     # Resize all images to uniform size 
            transforms.ToTensor(),                  # Convert image to PyTorch tensor
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image to a vector
        ])

        # Load dataset from folder structure (class_name/subfolder_name)
        self.image_dataset = datasets.ImageFolder(self.dataset_path, transform=transform)
   
        print(f"Dataset loaded with {len(self.image_dataset)} samples and {len(self.image_dataset.classes)} classes.")

    def separate_by_class(self):
        """Separate image features into class-specific NumPy arrays."""
        # Create an empty list for each class to store its image vectors
        self.class_features = {class_idx: [] for class_idx in range(len(self.image_dataset.classes))}
        # Loop through all images in the dataset
        for image, label in self.image_dataset:
            # Convert PyTorch tensor to NumPy array and add to corresponding class list
            self.class_features[label].append(image.numpy()) 
            # Convert each class's list of vectors into a single NumPy array
        for class_idx in self.class_features:
            self.class_features[class_idx] = np.array(self.class_features[class_idx])
            # Print the shape of the array: (number of images, number of features per image)
            print(f"[Class {class_idx}] -> Shape: {self.class_features[class_idx].shape}")


    def compute_centroid(self, class_idx: int):
        """Compute the mean vector (centroid) for a given class."""
        # Calculate the mean across all samples in the class (axis=0 = feature-wise mean)
        # This gives the "centroid" or average feature vector of the class
        return np.mean(self.class_features[class_idx], axis=0)

    def compute_spread(self, class_idx: int):
        """Compute the average spread (mean of standard deviations) for a given class."""
        # Compute standard deviation for each feature across all images in the class
        std_vector = np.std(self.class_features[class_idx], axis=0)
        # Take the mean of all feature stds to get a single scalar representing the class spread
        return np.mean(std_vector)
    
    def compute_centroid_distance(self, class_idx1: int, class_idx2: int):
        """Compute Euclidean distance between centroids of two classes."""
        # Compute the centroid (mean feature vector) of the first class
        centroid1 = self.compute_centroid(class_idx1)
        #  Compute the centroid of the second class
        centroid2 = self.compute_centroid(class_idx2)
        # Compute Euclidean distance between the two centroids
        # # This measures how far apart the two classes are in feature space
        return norm(centroid1 - centroid2)


    def evaluate(self, class_idx1=0, class_idx2=1):
        """Perform full evaluation for two given classes."""
        # Compute centroids for the two classes
        centroid_0 = self.compute_centroid(class_idx1)
        centroid_1 = self.compute_centroid(class_idx2)
        # Compute average spread (standard deviation) for each class
        spread_0 = self.compute_spread(class_idx1)
        spread_1 = self.compute_spread(class_idx2)
        # Compute Euclidean distance between the two class centroids
        distance = self.compute_centroid_distance(class_idx1, class_idx2)
        # Display the evaluation results
        print("\n[Evaluation Results]:")
        print(f"Class {class_idx1} Centroid Shape: {centroid_0.shape}")  # 1D vector size of features
        print(f"Class {class_idx2} Centroid Shape: {centroid_1.shape}")  # 1D vector size of features
        print(f"Class {class_idx1} Average Spread (Std Dev): {spread_0}")  # How diverse class 0 is
        print(f"Class {class_idx2} Average Spread (Std Dev): {spread_1}")  # How diverse class 1 is
        print(f"Distance Between Class Centroids: {distance}")            # How far apart the classes are


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Path to your dataset folder (should have subfolders for each class)
    dataset_path = r"C:\Users\divya\Desktop\labdataset"

    # Create an instance of the ClassSpreadEvaluator with the dataset path
    evaluator = ClassSpreadEvaluator(dataset_path)

    # Load the dataset and apply transformations (resize, flatten)
    evaluator.load_dataset()

    # Separate all images into class-specific NumPy arrays
    evaluator.separate_by_class()

    # Perform evaluation between class 0 and class 1
    # Computes centroids, spreads, and distance between centroids
    evaluator.evaluate(class_idx1=0, class_idx2=1)
