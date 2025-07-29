import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load and preprocess the dataset (resize, normalize, and flatten images) ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),                  # Resize all images to 84x84
    transforms.ToTensor(),                        # Convert PIL image to tensor
    transforms.Lambda(lambda img: img.view(-1))   # Flatten the image to a 1D vector
])

# Load the dataset from ImageFolder structure
image_dataset = datasets.ImageFolder(
   r'C:\Users\Divya\Desktop\Dataset',
    transform=image_transform
)

# --- 2. Extract flattened image vectors and their corresponding class labels ---
image_vectors = []  # To store flattened image data
image_labels = []   # To store labels (class indices)

for image_tensor, label in image_dataset:
    image_vectors.append(image_tensor.numpy())  # Convert tensor to NumPy array
    image_labels.append(label)

# Convert lists to NumPy arrays
image_vectors = np.array(image_vectors)
image_labels = np.array(image_labels)

# --- 3. Split the dataset into training and testing sets (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    image_vectors,
    image_labels,
    test_size=0.3,            # 30% data for testing
    random_state=42,          # For reproducibility
    stratify=image_labels     # Preserve class balance in split
)

# --- 4. Train a k-Nearest Neighbors (kNN) classifier with k=3 ---
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# --- 5. Make predictions on the test set ---
y_predicted = knn_classifier.predict(X_test)

# --- 6. Evaluate model performance ---
test_accuracy = accuracy_score(y_test, y_predicted)
print("Test Accuracy:", test_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_predicted, target_names=['Class 0', 'Class 1']))
