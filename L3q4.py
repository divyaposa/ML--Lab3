"""
A4. Divide dataset in your project into two parts – train & test set.
    - Use train_test_split() from scikit-learn.
    - X is the feature vector set, y is the class labels.
    - If dataset has multiple classes, choose only two for this task.
"""

import numpy as np  # for handling arrays
from torchvision import datasets, transforms  # for image dataset loading + transformations
from sklearn.model_selection import train_test_split  # for splitting train & test sets


class DatasetSplitter:
    """Class to load an image dataset, process features, and split into train/test sets."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        # save dataset path and image size (default = 84x84)
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.features = None  # will hold image data (X)
        self.labels = None    # will hold class labels (y)

    def load_dataset(self):
        """Load dataset with transformations and flatten images into vectors."""
        # define a set of transformations to apply to each image
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),     # resize image to 84x84
            transforms.ToTensor(),                 # convert image to tensor
            transforms.Lambda(lambda img: img.view(-1))  # flatten into 1D vector
        ])

        # load dataset from folders where each subfolder = one class
        dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)

        # extract features (images as vectors) and labels (class numbers)
        self.features = np.array([img.numpy() for img, _ in dataset])  # X
        self.labels = np.array([label for _, label in dataset])        # y

        print(f" Dataset loaded: {len(self.features)} images, {len(set(self.labels))} classes.")

    def ensure_two_classes(self):
        """If dataset has more than 2 classes, select only first two."""
        # get list of all unique class labels
        unique_classes = sorted(set(self.labels))
        if len(unique_classes) > 2:
            # if more than 2 classes, keep only first two
            print("⚠ Multi-class detected. Keeping only first two classes.")
            mask = np.isin(self.labels, unique_classes[:2])  # mask selects only first 2 classes
            self.features = self.features[mask]  # filter X
            self.labels = self.labels[mask]      # filter y

    def split_dataset(self, test_size=0.3):
        """Split dataset into training and testing sets."""
        # split into train (70%) and test (30%) while keeping class balance
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,        # X data (images)
            self.labels,          # y data (class labels)
            test_size=test_size,  # 30% test, 70% train
            random_state=42,      # fixed seed → results are repeatable
            stratify=self.labels  # ensures both sets keep same class ratio
        )

        # print details of split
        print(f"Total images: {len(self.features)}")
        print(f"Training set: {len(X_train)}")
        print(f" Testing set: {len(X_test)}")
        print(f" Class distribution (Train): {np.bincount(y_train)}")  # how many per class in train
        print(f"Class distribution (Test): {np.bincount(y_test)}")    # how many per class in test

        return X_train, X_test, y_train, y_test  # return split sets


# ----------- Main Execution -----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\divya\Desktop\labdataset"  # path where your dataset is stored

    splitter = DatasetSplitter(dataset_path)  # create object of DatasetSplitter
    splitter.load_dataset()                  # load and preprocess dataset
    splitter.ensure_two_classes()            # if more than 2 classes, keep only 2
    X_train, X_test, y_train, y_test = splitter.split_dataset(test_size=0.3)  # split into train/test
