"""
A6. Test the accuracy of the kNN using the test set obtained from the previous exercise (A5).
    - Load dataset and preprocess images
    - Split into train and test sets
    - Train kNN (k=3)
    - Evaluate accuracy, classification report, and confusion matrix
"""

import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class KNNTester:
    """Class to train and evaluate a kNN classifier on an image dataset."""

    def __init__(self, dataset_path, k=3, image_size=(84, 84)):
        self.dataset_path = dataset_path
        self.k = k
        self.image_size = image_size
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess(self):
        """Load dataset, flatten images, and split into train/test sets."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image
        ])

        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"✅ Data Loaded: {len(self.X_train)} train, {len(self.X_test)} test")

    def train_knn(self):
        """Train kNN classifier."""
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ kNN trained with k = {self.k}")

    def evaluate(self):
        """Evaluate kNN model and visualize confusion matrix."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"\n📊 Test Accuracy: {accuracy:.2f}")
        print("\n📄 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=["Class 0", "Class 1"]))

        cm = confusion_matrix(self.y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(cm)

        # Heatmap Visualization
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Class 0", "Class 1"],
                    yticklabels=["Class 0", "Class 1"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()


# ----------- Main Execution -----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset" 
    knn_tester = KNNTester(dataset_path, k=3)
    knn_tester.load_and_preprocess()
    knn_tester.train_knn()
    knn_tester.evaluate()
