from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from numpy.linalg import norm

# Define a transform to convert images to vectors
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # match miniImageNet size
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.view(-1))  # flatten the image
])

# Load the dataset
image_dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=transform)
data_loader = DataLoader(image_dataset, batch_size=600, shuffle=False)

# Initialize feature storage
class_0_features = []
class_1_features = []

# Load all data and separate by class
for image, label in image_dataset:
    if label == 0:
        class_0_features.append(image.numpy())
    else:
        class_1_features.append(image.numpy())

# Convert to NumPy arrays
class_0_features = np.array(class_0_features)  # shape: (N, flattened_size)
class_1_features = np.array(class_1_features)

# Compute mean vectors (centroids)
centroid_class_0 = np.mean(class_0_features, axis=0)
centroid_class_1 = np.mean(class_1_features, axis=0)

# Compute standard deviations (spread of each class)
std_dev_class_0 = np.std(class_0_features, axis=0)
std_dev_class_1 = np.std(class_1_features, axis=0)

# Average spread as scalar values
avg_spread_class_0 = np.mean(std_dev_class_0)
avg_spread_class_1 = np.mean(std_dev_class_1)

# Euclidean distance between class centroids
centroid_distance = norm(centroid_class_0 - centroid_class_1)

# Print results
print("Class 0 Centroid Shape:", centroid_class_0.shape)
print("Class 1 Centroid Shape:", centroid_class_1.shape)
print("Class 0 Average Spread (Std Dev):", avg_spread_class_0)
print("Class 1 Average Spread (Std Dev):", avg_spread_class_1)
print("Distance Between Class Centroids:", centroid_distance)
