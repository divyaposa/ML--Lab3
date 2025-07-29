import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and preprocess the MiniImageNet-style dataset (2 classes) ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),                # Resize images to 84x84 pixels
    transforms.ToTensor(),                      # Convert PIL images to PyTorch tensors
    transforms.Lambda(lambda img: img.view(-1)) # Flatten the image into a 1D vector
])

# Load images from the dataset folder (structured by class subfolders)
image_dataset = datasets.ImageFolder(
     r'C:\Users\Divya\Desktop\Dataset',
    transform=image_transform
)

# --- 2. Extract flattened image vectors (features) and their class labels ---
image_features = []  # List to hold flattened image vectors
image_labels = []    # List to hold corresponding class labels

for image_tensor, label in image_dataset:
    image_features.append(image_tensor.numpy())  # Convert tensor to NumPy array
    image_labels.append(label)

# Convert lists to NumPy arrays
image_features = np.array(image_features)
image_labels = np.array(image_labels)

# --- 3. Split data into training (70%) and testing (30%) sets ---
X_train, X_test, y_train, y_test = train_test_split(
    image_features,
    image_labels,
    test_size=0.3,
    random_state=42,
    stratify=image_labels  # Preserve class distribution
)

# --- 4. Train a k-Nearest Neighbors classifier with k=3 ---
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# --- 5. Predict class labels for the test set ---
y_predicted = knn_classifier.predict(X_test)

# --- 6. Evaluate the classifier using accuracy, classification report, and confusion matrix ---
test_accuracy = accuracy_score(y_test, y_predicted)
classification_summary = classification_report(y_test, y_predicted, target_names=['Class 0', 'Class 1'])
confusion_mat = confusion_matrix(y_test, y_predicted)

# --- 7. Print Evaluation Results ---
print("Test Accuracy: {:.2f}".format(test_accuracy))
print("\n Classification Report:\n", classification_summary)
print("\n Confusion Matrix:\n", confusion_mat)

# --- 8. Visualize Confusion Matrix as a heatmap ---
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
