"""
A2. Take any feature from your dataset.
    - Observe the density pattern for that feature by plotting the histogram.
    - Use buckets (data ranges) for histogram generation.
    - Calculate the mean and variance from the available data.

Steps:
    1. Load dataset and convert each image into a flattened feature vector.
    2. Select a specific feature index (pixel position in the flattened image).
    3. Extract values of that feature across all images.
    4. Calculate mean and variance of the feature values.
    5. Plot histogram using matplotlib to visualize density distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class FeatureHistogramAnalyzer:
    """Class to calculate mean, variance and plot histogram for a given dataset feature."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        """
        Initialize dataset path and transformation.

        Args:
            dataset_path (str): Path to the dataset in ImageFolder format.
            image_size (tuple): Size to resize images to (default 84x84 for MiniImageNet).
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.image_dataset = None
        self.flattened_features = None

    def load_dataset(self):
        """Load the dataset and flatten images to 1D feature vectors."""
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),        # Resize to uniform dimensions
            transforms.ToTensor(),                     # Convert to tensor
            transforms.Lambda(lambda img: img.view(-1))  # Flatten to 1D
        ])

        # Load dataset
        self.image_dataset = datasets.ImageFolder(self.dataset_path, transform=transform_pipeline)
        print(f"✅ Dataset loaded with {len(self.image_dataset)} images and {len(self.image_dataset.classes)} classes.")

    def extract_features(self):
        """Extract flattened image features into a NumPy array."""
        feature_list = [image.numpy() for image, _ in self.image_dataset]
        self.flattened_features = np.array(feature_list)
        print(f"📊 Feature matrix shape: {self.flattened_features.shape} (images x features)")

    def analyze_feature(self, feature_index: int, bins: int = 20):
        """
        Calculate mean, variance, and plot histogram for a given feature index.

        Args:
            feature_index (int): Index of the feature in the flattened vector.
            bins (int): Number of histogram bins (default 20).
        """
        if self.flattened_features is None:
            raise ValueError("Features not extracted. Call extract_features() first.")

        # Extract the selected feature values across all images
        feature_values = self.flattened_features[:, feature_index]

        # Calculate mean and variance
        mean_val = np.mean(feature_values)
        variance_val = np.var(feature_values)

        print(f"\n🔍 Feature Index: {feature_index}")
        print(f"📈 Mean: {mean_val}")
        print(f"📉 Variance: {variance_val}")

        # Plot histogram
        plt.hist(feature_values, bins=bins, edgecolor='black')
        plt.title(f"Histogram of Feature Index {feature_index}")
        plt.xlabel("Pixel Intensity (Normalized)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\Divya\Desktop\Dataset"   
    analyzer = FeatureHistogramAnalyzer(dataset_path)
    analyzer.load_dataset()
    analyzer.extract_features()
    analyzer.analyze_feature(feature_index=1000, bins=20)
