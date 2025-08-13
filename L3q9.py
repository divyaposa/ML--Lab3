"""
A9. Evaluate confusion matrix for your classification problem. 
    - From confusion matrix, compute precision, recall, and F1-score for both training and test data.
    - Based on observations, infer whether the model is underfit, regular fit, or overfit.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class KNNConfusionMatrixEvaluator:
    """Class to train k-NN and evaluate with confusion matrix and metrics."""

    def __init__(self, dataset_path, image_size=(84, 84), k=3):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.k = k
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k)

    def load_and_split_data(self):
        """Load dataset, preprocess, and split into train/test sets."""
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

        print(f"✅ Data Loaded: {len(self.X_train)} train samples, {len(self.X_test)} test samples")

    def train_model(self):
        """Train k-NN classifier."""
        self.knn_classifier.fit(self.X_train, self.y_train)

    def evaluate(self, X, y, dataset_name="Dataset"):
        """Evaluate model using accuracy, classification report, and confusion matrix."""
        predictions = self.knn_classifier.predict(X)

        # Accuracy
        accuracy = accuracy_score(y, predictions)

        # Classification report (Precision, Recall, F1-score)
        class_report = classification_report(
            y, predictions, target_names=['Class 0', 'Class 1']
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y, predictions)

        # Display results
        print(f"\n📊 {dataset_name} Accuracy: {accuracy:.4f}")
        print(f"🔍 {dataset_name} Classification Report:\n{class_report}")
        print(f"🧮 {dataset_name} Confusion Matrix:\n{conf_matrix}")

        # Plot confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1']
        )
        plt.title(f"{dataset_name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.show()

        return accuracy

    def run_full_evaluation(self):
        """Run the complete evaluation process."""
        self.load_and_split_data()
        self.train_model()

        print("\n===== 🏋️ Training Set Evaluation =====")
        train_acc = self.evaluate(self.X_train, self.y_train, "Training Set")

        print("\n===== 🧪 Test Set Evaluation =====")
        test_acc = self.evaluate(self.X_test, self.y_test, "Test Set")

        # Model fit assessment
        if train_acc > 0.98 and test_acc < (train_acc - 0.1):
            fit_status = "Overfit"
        elif abs(train_acc - test_acc) <= 0.05:
            fit_status = "Regular Fit"
        else:
            fit_status = "Underfit"

        print(f"\n📌 Model Fit Assessment: {fit_status}")


# ---------- Main Execution ----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"  
    evaluator = KNNConfusionMatrixEvaluator(dataset_path, k=3)
    evaluator.run_full_evaluation()

