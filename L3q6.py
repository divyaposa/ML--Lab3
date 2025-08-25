"""
A6. Test the accuracy of the kNN using the test set obtained from the previous exercise (A5).
    - Load dataset and preprocess images
    - Split into train and test sets
    - Train kNN (k=3)
    - Evaluate accuracy, classification report, and confusion matrix
"""

import numpy as np  # For handling arrays
from torchvision import datasets, transforms  # For loading and transforming image dataset
from sklearn.model_selection import train_test_split  # To split dataset into train/test sets
from sklearn.neighbors import KNeighborsClassifier  # kNN classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For evaluation metrics
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For confusion matrix heatmap


class KNNTester:
    """Class to train and evaluate a kNN classifier on an image dataset."""

    def __init__(self, dataset_path, k=3, image_size=(84, 84)):
        # Save dataset path, k value, and image size
        self.dataset_path = dataset_path
        self.k = k
        self.image_size = image_size
        self.model = None  # Placeholder for kNN model
        self.X_train = None  # Training data (features)
        self.X_test = None   # Test data (features)
        self.y_train = None  # Training labels
        self.y_test = None   # Test labels

    def load_and_preprocess(self):
        """Load dataset, flatten images, and split into train/test sets."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),   # Resize all images to fixed size (84x84)
            transforms.ToTensor(),                # Convert image to PyTorch tensor
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image into a 1D vector
        ])

        # Load dataset from folder (expects subfolders per class)
        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)

        # Convert dataset into NumPy arrays: features = images, labels = class numbers
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        # Split dataset into train (70%) and test (30%) sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f" Data Loaded: {len(self.X_train)} train, {len(self.X_test)} test")

    def train_knn(self):
        """Train kNN classifier."""
        self.model = KNeighborsClassifier(n_neighbors=self.k)  # Create kNN model with chosen k
        self.model.fit(self.X_train, self.y_train)  # Train model using training data
        print(f" kNN trained with k = {self.k}")

    def evaluate(self):
        """Evaluate kNN model and visualize confusion matrix."""
        y_pred = self.model.predict(self.X_test)  # Predict labels for test data
        accuracy = accuracy_score(self.y_test, y_pred)  # Calculate accuracy

        print(f"\n Test Accuracy: {accuracy:.2f}")
        print("\n Classification Report:")
        # Show precision, recall, f1-score for each class
        print(classification_report(self.y_test, y_pred, target_names=["Class 0", "Class 1"]))

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot confusion matrix as heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Class 0", "Class 1"],  # Predicted labels
                    yticklabels=["Class 0", "Class 1"])  # Actual labels
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()


# ----------- Main Execution -----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\divya\Desktop\labdataset"  # Path to dataset
    knn_tester = KNNTester(dataset_path, k=3)  # Create tester object with k=3
    knn_tester.load_and_preprocess()  # Load and split dataset
    knn_tester.train_knn()            # Train kNN model
    knn_tester.evaluate()             # Test model and show results
