import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Define image preprocessing pipeline ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),                  # Resize image to 84x84 pixels
    transforms.ToTensor(),                        # Convert image to PyTorch tensor
    transforms.Lambda(lambda img: img.view(-1))   # Flatten image into 1D vector
])

# --- 2. Load dataset from directory (structured with subfolders for each class) ---
image_dataset = datasets.ImageFolder(
    r'C:\Users\Divya\Desktop\Dataset',
    transform=image_transform
)

# --- 3. Extract features (X) and labels (y) ---
image_features = []  # List of flattened image vectors
image_labels = []    # List of corresponding class labels

for image_tensor, label in image_dataset:
    image_features.append(image_tensor.numpy())
    image_labels.append(label)

# Convert to NumPy arrays
image_features = np.array(image_features)
image_labels = np.array(image_labels)

# --- 4. Split dataset into training and testing sets (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    image_features,
    image_labels,
    test_size=0.3,
    random_state=42,
    stratify=image_labels  # Preserve class balance
)

# --- 5. Test multiple k values for k-NN and record accuracy ---
k_values = list(range(1, 12))  # Try k from 1 to 11
k_accuracies = []              # Store accuracy for each k

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    y_predicted = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    
    k_accuracies.append(accuracy)
    print(f"k = {k}, Accuracy = {accuracy:.4f}")

# --- 6. Plot Accuracy vs k ---
plt.figure(figsize=(8, 5))
plt.plot(k_values, k_accuracies, marker='o', linestyle='-')
plt.title('k-NN Accuracy vs k (MiniImageNet 2-Class)')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
