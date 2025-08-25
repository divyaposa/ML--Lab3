"""
A3. Take any two feature vectors from your dataset.
    - Calculate the Minkowski distance for r from 1 to 10.
    - Make a plot of the distance and observe the nature of the graph.
"""

import numpy as np   # Import NumPy library for numerical operations and handling arrays
import matplotlib.pyplot as plt   # Import Matplotlib for plotting graphs and visualizing data
from torchvision import datasets, transforms   # Import datasets (like MNIST, CIFAR10) and transforms for preprocessing images
from sklearn.model_selection import train_test_split   # Import function to split data into training and testing sets


class MinkowskiDistanceAnalyzer:
    """Class to compute Minkowski distances between two feature vectors and split dataset."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        # Save dataset path (where images are stored)
        self.dataset_path = dataset_path
        # Save target image size (resize all images to this size, default 84x84)
        self.image_size = image_size
        # Placeholder for dataset (will store images + labels after loading)
        self.dataset = None
        # Placeholder for extracted feature vectors (numerical data from images)
        self.features = None
        # Placeholder for labels (class/category of each image)
        self.labels = None


    def load_dataset(self):
        """Load dataset and flatten images into feature vectors."""
        # Define a pipeline of transformations applied to each image
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),          # Resize image to given size (e.g., 84x84 pixels)
            transforms.ToTensor(),                       # Convert image into a PyTorch tensor (values in [0,1])
            transforms.Lambda(lambda img: img.view(-1))  # Flatten the tensor into a 1D vector (feature vector) 
            ])
        # Load the dataset from the folder path where images are stored
        # # ImageFolder expects subfolders as class labels (e.g., 'cats/', 'dogs/')
        self.dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        # Extract features and labels from the dataset
        self.features = np.array([img.numpy() for img, _ in self.dataset])  # Convert each image to a NumPy array and store in feature
        self.labels = np.array([label for _, label in self.dataset])         # Collect all labels from the dataset
        # Print how many images and classes are loaded
        print(f" Loaded {len(self.dataset)} images across {len(set(self.labels))} classes.")  # len(set(...)) gives number of unique classes


    def compute_minkowski_distances(self, vector1_idx=0, vector2_idx=1, r_max=10):
        """Compute Minkowski distances between two selected feature vectors.
        Args:
        vector1_idx (int): Index of first vector.
        vector2_idx (int): Index of second vector.
        r_max (int): Maximum r value (default 10).
        """
        vec1 = self.features[vector1_idx]  # Get the first feature vector by its index
        vec2 = self.features[vector2_idx]  # Get the second feature vector by its index
        r_values = list(range(1, r_max + 1))  # Create a list of r values from 1 to r_max
        distances = [
            np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r)  # Compute Minkowski distance for each r
            for r in r_values
            ]
        # Plot distances
        plt.plot(r_values, distances, marker='o')   # Plot r values on x-axis and distances on y-axis with circle markers
        plt.title("Minkowski Distance vs r")        # Add title to the graph
        plt.xlabel("r (Minkowski Power Parameter)") # Label x-axis
        plt.ylabel("Distance")                      # Label y-axis
        plt.grid(True)                              # Show grid for better readability
        plt.show()                                  # Display the plot
        # Print results
        for r, dist in zip(r_values, distances):    # Loop through each r and its corresponding distance
            print(f"r = {r}: Distance = {dist:.4f}") # Print r value and distance rounded to 4 decimal places


    def split_dataset(self, test_size=0.3):
        """
        Split dataset into train and test sets (only two classes).
        Args:
        test_size (float): Proportion for test split (default 0.3).
        """
        # Get all unique classes in the dataset
        unique_classes = sorted(set(self.labels))
        # If more than 2 classes, select only the first two
        if len(unique_classes) > 2:
            print("Multi-class dataset detected. Selecting only first two classes.")
            mask = np.isin(self.labels, unique_classes[:2])  # Create a mask for first two classes
            X = self.features[mask]  # Select feature vectors of first two classes
            y = self.labels[mask]    # Select labels of first two classes
        else:
            X = self.features  # Use all features if only two classes
            y = self.labels    # Use all labels if only two classes
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42  # random_state for reproducibility
                  )
            # Print number of samples in each set
            print(f"Train set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            return X_train, X_test, y_train, y_test  # Return the splits


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\divya\Desktop\labdataset"  # Path to your dataset folder

    # Create an instance of the MinkowskiDistanceAnalyzer class
    analyzer = MinkowskiDistanceAnalyzer(dataset_path)
    
    # Load images and labels from the dataset
    analyzer.load_dataset()
    
    # Compute Minkowski distances between first two feature vectors
    analyzer.compute_minkowski_distances(vector1_idx=0, vector2_idx=1, r_max=10)

    # Split the dataset into training and testing sets (30% test by default)
    X_train, X_test, y_train, y_test = analyzer.split_dataset(test_size=0.3)
