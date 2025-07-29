import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# --- 1. Load dataset with image transformation and flattening ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.view(-1))  # flatten the image
])

image_dataset = datasets.ImageFolder(r'C:\Users\<your-username>\Desktop\Dataset', transform=image_transform)

# --- 2. Extract image features and corresponding labels ---
image_features = []
image_labels = []

for image_tensor, label in image_dataset:
    image_features.append(image_tensor.numpy())
    image_labels.append(label)

image_features = np.array(image_features)  # shape: (num_images, flattened_vector_size)
image_labels = np.array(image_labels)      # shape: (num_images,)

# --- 3. Split into training and testing sets (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    image_features,
    image_labels,
    test_size=0.3,
    random_state=42,
    stratify=image_labels
)

# --- 4. Print dataset sizes and class distributions ---
print("Total number of images:", len(image_features))
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

print("Class distribution in training set:", np.bincount(y_train))
print("Class distribution in testing set:", np.bincount(y_test))
