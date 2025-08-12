"""
A4. Divide dataset in your project into two parts – train & test set.
    - Use train_test_split() from scikit-learn.
    - X is the feature vector set, y is the class labels.
    - If dataset has multiple classes, choose only two for this task.
"""

import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Class to load an image dataset, process features, and split into train/test sets."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.features = None
        self.labels = None

    def load_dataset(self):
        """Load dataset with transformations and flatten images into vectors."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.view(-1))  # flatten image
        ])

        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)

        # Extract features and labels
        self.features = np.array([img.numpy() for img, _ in dataset])
        self.labels = np.array([label for _, label in dataset])

        print(f"✅ Dataset loaded: {len(self.features)} images, {len(set(self.labels))} classes.")

    def ensure_two_classes(self):
        """If dataset has more than 2 classes, select only first two."""
        unique_classes = sorted(set(self.labels))
        if len(unique_classes) > 2:
            print("⚠ Multi-class detected. Keeping only first two classes.")
            mask = np.isin(self.labels, unique_classes[:2])
            self.features = self.features[mask]
            self.labels = self.labels[mask]

    def split_dataset(self, test_size=0.3):
        """Split dataset into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            random_state=42,
            stratify=self.labels
        )

        # Print dataset details
        print(f"📊 Total images: {len(self.features)}")
        print(f"📊 Training set: {len(X_train)}")
        print(f"📊 Testing set: {len(X_test)}")
        print(f"📊 Class distribution (Train): {np.bincount(y_train)}")
        print(f"📊 Class distribution (Test): {np.bincount(y_test)}")

        return X_train, X_test, y_train, y_test


# ----------- Main Execution -----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"   

    splitter = DatasetSplitter(dataset_path)
    splitter.load_dataset()
    splitter.ensure_two_classes()
    X_train, X_test, y_train, y_test = splitter.split_dataset(test_size=0.3)
