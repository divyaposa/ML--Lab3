"""
A7. Use the predict() function to study the prediction behavior of the classifier for test vectors.

    - Train kNN classifier using the train set from A6
    - Predict on the entire test set
    - Compare predicted and actual labels for a few samples
    - Manually test prediction for a specific vector
"""

# Import libraries
import numpy as np                           # For handling arrays
from torchvision import datasets, transforms # For loading and transforming image dataset
from sklearn.model_selection import train_test_split # For splitting train and test sets
from sklearn.neighbors import KNeighborsClassifier   # kNN model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # For evaluation
import matplotlib.pyplot as plt              # For plotting graphs
import seaborn as sns                        # For heatmap visualization


class KNNPredictionStudy:
    """Class to train, evaluate, and demonstrate predictions of a kNN classifier."""

    def __init__(self, dataset_path, k=3, image_size=(84, 84)):
        # Save input values (dataset path, k, image size)
        self.dataset_path = dataset_path
        self.k = k
        self.image_size = image_size
        self.model = None     # Placeholder for the kNN model
        self.X_train = None   # Training features
        self.X_test = None    # Testing features
        self.y_train = None   # Training labels
        self.y_test = None    # Testing labels

    def load_and_split_data(self):
        """Load dataset, preprocess, and split into training/testing sets."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),   # Resize all images to same size
            transforms.ToTensor(),                # Convert image to tensor (numerical format)
            transforms.Lambda(lambda img: img.view(-1))  # Flatten image into 1D vector
        ])

        # Load dataset from folders (expects subfolders as classes)
        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)

        # Extract features (flattened images) and labels (class indices)
        features = np.array([img.numpy() for img, _ in dataset])
        labels = np.array([label for _, label in dataset])

        # Split into training (70%) and testing (30%) sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f" Data Loaded: {len(self.X_train)} train, {len(self.X_test)} test")

    def train_model(self):
        """Train the kNN classifier."""
        self.model = KNeighborsClassifier(n_neighbors=self.k) # Create kNN with given k
        self.model.fit(self.X_train, self.y_train)            # Train using training data
        print(f"kNN trained with k = {self.k}")

    def evaluate(self):
        """Evaluate the model's performance on the test set."""
        predictions = self.model.predict(self.X_test)   # Predict classes for all test samples
        accuracy = accuracy_score(self.y_test, predictions) # Calculate accuracy

        print(f"\nTest Accuracy: {accuracy:.2f}")       # Print accuracy
        print("\n Classification Report:")
        print(classification_report(self.y_test, predictions, target_names=["Class 0", "Class 1"]))

        # Create confusion matrix (comparison of predicted vs actual)
        cm = confusion_matrix(self.y_test, predictions)
        print("\nConfusion Matrix:\n", cm)

        # Show confusion matrix as heatmap (colored grid)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Class 0", "Class 1"],  # Columns = predicted
                    yticklabels=["Class 0", "Class 1"])  # Rows = actual
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    def show_sample_predictions(self, sample_count=10):
        """Display predictions vs actual labels for the first N test samples."""
        print(f"\n First {sample_count} Predictions vs Actual Labels:")
        for i in range(sample_count):                         # Loop through first N test samples
            predicted_class = self.model.predict([self.X_test[i]])[0] # Predict class
            actual_class = self.y_test[i]                     # True class
            print(f"Test Sample {i}: Predicted = Class {predicted_class}, Actual = Class {actual_class}")

    def predict_single_sample(self, index):
        """Predict class for a specific test vector."""
        test_vector = self.X_test[index]                      # Pick test sample at given index
        predicted_class = self.model.predict([test_vector])[0] # Predict its class
      
        print(f"\n Prediction for test sample {index}: Predicted = Class {predicted_class}, Actual = Class {self.y_test[index]}")



# ---------- Main Execution ----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\divya\Desktop\labdataset"  # Path to your dataset

    # Create object of class
    knn_study = KNNPredictionStudy(dataset_path, k=3)
    knn_study.load_and_split_data()        # Load and preprocess dataset
    knn_study.train_model()                # Train kNN model
    knn_study.evaluate()                   # Check accuracy and confusion matrix
    knn_study.show_sample_predictions(sample_count=10) # Show first 10 predictions
    knn_study.predict_single_sample(index=50)          # Predict specific sample
