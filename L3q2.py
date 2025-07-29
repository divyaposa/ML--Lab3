import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Load the dataset with transformations ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.view(-1))  # flatten to 1D vector
])

image_dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=image_transform)

# --- 2. Extract flattened image features into a NumPy array ---
flattened_image_features = []
for image_tensor, label in image_dataset:
    flattened_image_features.append(image_tensor.numpy())

flattened_image_features = np.array(flattened_image_features)  # shape: (num_images, flattened_size)

# --- 3. Select a specific feature index ---
target_feature_index = 1000
feature_values = flattened_image_features[:, target_feature_index]  # shape: (num_images,)

# --- 4. Calculate Mean and Variance of the selected feature ---
feature_mean = np.mean(feature_values)
feature_variance = np.var(feature_values)

print(f"Feature Index: {target_feature_index}")
print(f"Mean: {feature_mean}")
print(f"Variance: {feature_variance}")

# --- 5. Plot Histogram of the selected feature values ---
plt.hist(feature_values, bins=20, edgecolor='black')
plt.title(f'Histogram of Feature Index {target_feature_index}')
plt.xlabel('Pixel Intensity (Normalized)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
