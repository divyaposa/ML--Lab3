"""
A5. Train a kNN classifier (k = 3) using the training set from the previous exercise (A4).
    - Load the dataset
    - Flatten the image vectors
    - Split into train/test sets
    - Train a kNN classifier
    - Evaluate accuracy and classification report
"""

# Import necessary libraries
import numpy as np
from torchvision import datasets, transforms   # for loading and transforming images
from sklearn.model_selection import train_test_split  # for splitting dataset
from sklearn.neighbors import KNeighborsClassifier    # kNN classifier
from sklearn.metrics import accuracy_score, classification_report  # evaluation metrics


class KNNImageClassifier:
    """A class to handle dataset loading, preprocessing, and kNN classification."""

    def __init__(self, dataset_path, image_size=(84, 84), k=3):
        # Save the dataset path (folder where images are stored)
        self.dataset_path = dataset_path
        # Resize all images to this size (84x84 pixels by default)
        self.image_size = image_size
        # Number of neighbors for kNN
        self.k = k
        # Placeholders for train/test data and labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # Placeholder for trained kNN model
        self.model = None

    def load_and_preprocess_data(self):
        """Load dataset and flatten images."""
        # Define how to process each image:
        # 1. Resize to 84x84
        # 2. Convert to tensor (PyTorch format)
        # 3. Flatten to a single row vector
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image to 1D vector
        ])

        # Load dataset from given path
        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)

        # Convert dataset into numpy arrays: features (X) and labels (y)
        features = np.array([img.numpy() for img, _ in dataset])   # image vectors
        labels = np.array([label for _, label in dataset])         # class labels

        # If dataset has more than 2 classes, keep only the first two
        unique_classes = sorted(set(labels))
        if len(unique_classes) > 2:
            print("⚠ Multi-class detected. Keeping only first two classes.")
            mask = np.isin(labels, unique_classes[:2])   # keep only class 0 and 1
            features, labels = features[mask], labels[mask]

        # Split data into training (70%) and testing (30%) sets
        # stratify ensures class balance is maintained in both sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"Data split: {len(self.X_train)} train, {len(self.X_test)} test")

    def train_knn(self):
        """Train kNN classifier."""
        # Create kNN classifier with chosen k (number of neighbors)
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        # Train model using training data
        self.model.fit(self.X_train, self.y_train)
        print(f" kNN classifier trained with k = {self.k}")

    def evaluate_model(self):
        """Evaluate the trained kNN model."""
        # Predict labels for test set
        predictions = self.model.predict(self.X_test)
        # Calculate overall accuracy
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        # Print precision, recall, f1-score for each class
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions, target_names=["Class 0", "Class 1"]))


# ----------- Main Execution -----------
if __name__ == "__main__":
    # Path to dataset (change it to your dataset folder)
    dataset_path = r"C:\Users\divya\Desktop\labdataset" 

    # Create object of the KNNImageClassifier class with k=3
    knn_classifier = KNNImageClassifier(dataset_path, k=3)
    # Step 1: Load and preprocess the data
    knn_classifier.load_and_preprocess_data()
    # Step 2: Train the kNN classifier
    knn_classifier.train_knn()
    # Step 3: Evaluate the trained model
    knn_classifier.evaluate_model()
