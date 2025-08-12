"""
A5. Train a kNN classifier (k = 3) using the training set from the previous exercise (A4).
    - Load the dataset
    - Flatten the image vectors
    - Split into train/test sets
    - Train a kNN classifier
    - Evaluate accuracy and classification report
"""

import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


class KNNImageClassifier:
    """A class to handle dataset loading, preprocessing, and kNN classification."""

    def __init__(self, dataset_path, image_size=(84, 84), k=3):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.k = k
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_and_preprocess_data(self):
        """Load dataset and flatten images."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image
        ])

        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        # Keep only two classes if multi-class
        unique_classes = sorted(set(labels))
        if len(unique_classes) > 2:
            print("⚠ Multi-class detected. Keeping only first two classes.")
            mask = np.isin(labels, unique_classes[:2])
            features, labels = features[mask], labels[mask]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"✅ Data split: {len(self.X_train)} train, {len(self.X_test)} test")

    def train_knn(self):
        """Train kNN classifier."""
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ kNN classifier trained with k = {self.k}")

    def evaluate_model(self):
        """Evaluate the trained kNN model."""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"\n📊 Test Accuracy: {accuracy:.4f}")
        print("\n📑 Classification Report:")
        print(classification_report(self.y_test, predictions, target_names=["Class 0", "Class 1"]))


# ----------- Main Execution -----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"  

    knn_classifier = KNNImageClassifier(dataset_path, k=3)
    knn_classifier.load_and_preprocess_data()
    knn_classifier.train_knn()
    knn_classifier.evaluate_model()
