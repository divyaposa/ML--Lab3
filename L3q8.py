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
        # Save dataset location and image size
        self.dataset_path = dataset_path
        self.image_size = image_size
        
        # Train/test sets (empty for now)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # List of k values from 1 to 11
        self.k_values = list(range(1, 12))
        
        # Store accuracy for each k
        self.k_accuracies = []

    def load_and_split_data(self):
        """Load dataset, preprocess images, and split into train/test sets."""
        
        # Steps to prepare images:
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),       # Resize each image
            transforms.ToTensor(),                   # Convert image to tensor
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image into 1D vector
        ])

        # Load dataset from folder with labels
        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        
        # Convert all images and labels to numpy arrays
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        # Split data into training (70%) and testing (30%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f" Data Loaded: {len(self.X_train)} train, {len(self.X_test)} test")

    def evaluate_for_k_values(self):
        """Train k-NN for different k values and record accuracy."""
        
        # Loop through k = 1, 2, ..., 11
        for k in self.k_values:
            # Create k-NN model with k neighbors
            knn = KNeighborsClassifier(n_neighbors=k)
            
            # Train the model using training data
            knn.fit(self.X_train, self.y_train)

            # Predict labels on the test set
            predictions = knn.predict(self.X_test)
            
            # Calculate accuracy of predictions
            accuracy = accuracy_score(self.y_test, predictions)

            # Save accuracy for plotting later
            self.k_accuracies.append(accuracy)
            
            # Print result for this k
            print(f"k = {k}, Accuracy = {accuracy:.4f}")

    def plot_accuracy_vs_k(self):
        """Plot accuracy scores for different k values."""
        
        # Create a line plot of accuracy vs k
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_values, self.k_accuracies, marker='o', linestyle='-')
        plt.title('k-NN Accuracy vs k')
        plt.xlabel('k (Number of Neighbors)')
        plt.ylabel('Accuracy')
        plt.xticks(self.k_values)  # Show all k values on x-axis
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------- Main Execution ----------
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = r"C:\Users\divya\Desktop\labdataset"   

    # Create object of the class
    knn_comparison = KNNKComparison(dataset_path)
    
    # Load data and split into train/test
    knn_comparison.load_and_split_data()
    
    # Train and evaluate model for different k values
    knn_comparison.evaluate_for_k_values()
    
    # Plot accuracy vs k
    knn_comparison.plot_accuracy_vs_k()
