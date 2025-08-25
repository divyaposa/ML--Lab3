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

import numpy as np                      # Import NumPy library for numerical operations (like mean, variance, histogram)
import matplotlib.pyplot as plt         # Import Matplotlib for plotting graphs (like histogram)
from torchvision import datasets, transforms  # Import torchvision tools (datasets and transforms for image data handling)



class FeatureHistogramAnalyzer:
    """Class to calculate mean, variance and plot histogram for a given dataset feature."""

    def __init__(self, dataset_path: str, image_size=(84, 84)):
        """
        Initialize dataset path and transformation.

        Args:
            dataset_path (str): Path to the dataset in ImageFolder format.
            image_size (tuple): Size to resize images to (default 84x84 for MiniImageNet).
        """
        self.dataset_path = dataset_path       # Store the dataset path
        self.image_size = image_size           # Store the image resize size (default 84x84)
        self.image_dataset = None              # Will later hold the dataset after loading
        self.flattened_features = None         # Will later hold pixel values (features) in 1D form


    def load_dataset(self):
        """Load the dataset and flatten images to 1D feature vectors."""
        # Define a pipeline of transformations applied to each image
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.image_size),          # Resize all images to the given size (e.g., 64x64)
            transforms.ToTensor(),                       # Convert image (PIL/numpy) into a PyTorch tensor [0,1]
            transforms.Lambda(lambda img: img.view(-1))  # Flatten the 2D/3D tensor into a 1D vector for easier processing
            ])
        # Load dataset from the given folder path using ImageFolder
        # ImageFolder expects data in the format: root/class_name/*.jpg (or png etc.)
        self.image_dataset = datasets.ImageFolder(
            self.dataset_path, 
            transform=transform_pipeline
            )
        # Print a summary of dataset information
        print(
            f"Dataset loaded with {len(self.image_dataset)} images "
            f"and {len(self.image_dataset.classes)} classes."
            )


    def extract_features(self):
        """Extract flattened image features into a NumPy array."""
        # Go through each image in the dataset and convert the tensor to a NumPy array
        # `image` is the transformed image, `_` is the label (ignored here)
        feature_list = [image.numpy() for image, _ in self.image_dataset]
        # Convert the list of NumPy arrays into a single 2D NumPy array
        # # Shape will be: (number_of_images, number_of_features_per_image)
        self.flattened_features = np.array(feature_list)
        # Print the shape of the feature matrix for confirmation
        print(f"Feature matrix shape: {self.flattened_features.shape} (images x features)")


    def analyze_feature(self, feature_index: int, bins: int = 20):
        """Calculate mean, variance, and plot histogram for a given feature index.
        Args:
        feature_index (int): Index of the feature in the flattened vector.
        bins (int): Number of histogram bins (default 20)."""
        # Check if features have been extracted before analysis
        if self.flattened_features is None:
            raise ValueError("Features not extracted. Call extract_features() first.")
        # Get all values of the selected feature across all images
        feature_values = self.flattened_features[:, feature_index]
        # Calculate the mean of this feature (average intensity)
        mean_val = np.mean(feature_values)
        # Calculate the variance of this feature (spread of values)
        variance_val = np.var(feature_values)
        # Print feature statistics
        print(f"\nFeature Index: {feature_index}")
        print(f"Mean: {mean_val}")
        print(f"Variance: {variance_val}")
         # Plot histogram of the feature values
        plt.hist(feature_values, bins=bins, edgecolor='black')   # histogram with bin count
        plt.title(f"Histogram of Feature Index {feature_index}") # title of histogram
        plt.xlabel("Pixel Intensity (Normalized)")               # x-axis label
        plt.ylabel("Frequency")                                  # y-axis label
        plt.grid(True)                                           # enable grid for readability
        plt.show()                                               # display the plot



# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Path to the dataset folder (update this if your dataset is elsewhere)
    dataset_path = r"C:\Users\divya\Desktop\labdataset"  
    
    # Create an analyzer object with the dataset path
    analyzer = FeatureHistogramAnalyzer(dataset_path)
    
    # Load the dataset (images + labels) from the given path
    analyzer.load_dataset()
    
    # Extract features by flattening images into 1D arrays
    analyzer.extract_features()
    
    # Analyze a specific feature (e.g., pixel index 1000) with 20 bins in the histogram
    analyzer.analyze_feature(feature_index=1000, bins=20)

