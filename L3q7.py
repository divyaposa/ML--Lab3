"""
A7. Use the predict() function to study the prediction behavior of the classifier for test vectors.

    - Train kNN classifier using the train set from A6
    - Predict on the entire test set
    - Compare predicted and actual labels for a few samples
    - Manually test prediction for a specific vector
"""

import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class KNNPredictionStudy:
    """Class to train, evaluate, and demonstrate predictions of a kNN classifier."""

    def __init__(self, dataset_path, k=3, image_size=(84, 84)):
        self.dataset_path = dataset_path
        self.k = k
        self.image_size = image_size
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_split_data(self):
        """Load dataset, preprocess, and split into training/testing sets."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))  # Flatten
        ])

        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"✅ Data Loaded: {len(self.X_train)} train, {len(self.X_test)} test")

    def train_model(self):
        """Train the kNN classifier."""
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ kNN trained with k = {self.k}")

    def evaluate(self):
        """Evaluate the model's performance on the test set."""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)

        print(f"\n📊 Test Accuracy: {accuracy:.2f}")
        print("\n📄 Classification Report:")
        print(classification_report(self.y_test, predictions, target_names=["Class 0", "Class 1"]))

        cm = confusion_matrix(self.y_test, predictions)
        print("\n🔢 Confusion Matrix:\n", cm)

        # Heatmap visualization
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Class 0", "Class 1"],
                    yticklabels=["Class 0", "Class 1"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    def show_sample_predictions(self, sample_count=10):
        """Display predictions vs actual labels for the first N test samples."""
        print(f"\n🔍 First {sample_count} Predictions vs Actual Labels:")
        for i in range(sample_count):
            predicted_class = self.model.predict([self.X_test[i]])[0]
            actual_class = self.y_test[i]
            print(f"Test Sample {i}: Predicted = Class {predicted_class}, Actual = Class {actual_class}")

    def predict_single_sample(self, index):
        """Predict class for a specific test vector."""
        test_vector = self.X_test[index]
        predicted_class = self.model.predict([test_vector])[0]
        print(f"\n🎯 Prediction for test sample {index} → Class {predicted_class}, Actual → Class {self.y_test[index]}")


# ---------- Main Execution ----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"  

    knn_study = KNNPredictionStudy(dataset_path, k=3)
    knn_study.load_and_split_data()
    knn_study.train_model()
    knn_study.evaluate()
    knn_study.show_sample_predictions(sample_count=10)
    knn_study.predict_single_sample(index=50)
