import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Image preprocessing: resize, convert to tensor, flatten ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),                  # Resize to 84x84
    transforms.ToTensor(),                        # Convert to PyTorch tensor
    transforms.Lambda(lambda img: img.view(-1))   # Flatten image to 1D vector
])

# --- 2. Load dataset from folder ---
image_dataset = datasets.ImageFolder(
   r'C:\Users\Divya\Desktop\Dataset',
    transform=image_transform
)

# --- 3. Extract image features and labels ---
image_features = []  # Stores flattened image vectors
image_labels = []    # Stores corresponding class labels

for image_tensor, label in image_dataset:
    image_features.append(image_tensor.numpy())  # Convert tensor to NumPy
    image_labels.append(label)

# Convert to NumPy arrays for ML model use
image_features = np.array(image_features)
image_labels = np.array(image_labels)

# --- 4. Split into training and testing sets (70/30) ---
X_train, X_test, y_train, y_test = train_test_split(
    image_features,
    image_labels,
    test_size=0.3,
    random_state=42,
    stratify=image_labels  # Preserve class distribution
)

# --- 5. Initialize and train k-NN classifier (k = 3) ---
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# --- 6. Predict on test set ---
predicted_labels = knn_classifier.predict(X_test)

# --- 7. Evaluate the model ---
test_accuracy = accuracy_score(y_test, predicted_labels)
print("Test Accuracy: {:.2f}".format(test_accuracy))

print("\n Classification Report:\n", classification_report(y_test, predicted_labels))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, predicted_labels))

# --- 8. Visualize Confusion Matrix ---
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, predicted_labels), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ==========================
# === A7: Prediction Task ===
# ==========================

# --- 9. Show predictions for the first 10 test samples ---
print("\n First 10 Predictions vs Actual Labels:")
for i in range(10):
    predicted_class = knn_classifier.predict([X_test[i]])[0]
    actual_class = y_test[i]
    print(f"Test Sample {i}: Predicted = Class {predicted_class}, Actual = Class {actual_class}")

# --- 10. Predict a single sample manually (e.g., sample 50) ---
print("\n Predicting one test vector directly:")
manual_test_vector = X_test[50]
manual_predicted_class = knn_classifier.predict([manual_test_vector])[0]
print(f"Prediction for test sample 50 → Class {manual_predicted_class}, Actual → Class {y_test[50]}")
