import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define preprocessing pipeline (resize, tensor, flatten) ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),                  # Resize to 84x84 as per MiniImageNet format
    transforms.ToTensor(),                        # Convert PIL image to PyTorch tensor
    transforms.Lambda(lambda img: img.view(-1))   # Flatten image into a 1D vector
])

# --- 2. Load image dataset from directory ---
image_dataset = datasets.ImageFolder(
    r'C:\Users\Divya\Desktop\Dataset',
    transform=image_transform
)

# --- 3. Extract image vectors (features) and labels ---
image_features = []  # To store flattened image arrays
image_labels = []    # To store corresponding class labels

for image_tensor, label in image_dataset:
    image_features.append(image_tensor.numpy())  # Convert tensor to NumPy array
    image_labels.append(label)

# Convert to NumPy arrays for model input
image_features = np.array(image_features)
image_labels = np.array(image_labels)

# --- 4. Split into training and testing sets (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    image_features,
    image_labels,
    test_size=0.3,
    random_state=42,
    stratify=image_labels  # Maintain class balance
)

# --- 5. Initialize and train kNN classifier (k = 3) ---
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# --- 6. Predict on both training and testing data ---
train_predictions = knn_classifier.predict(X_train)
test_predictions = knn_classifier.predict(X_test)

# --- 7. Define evaluation function for classification metrics & confusion matrix ---
def evaluate_classifier(true_labels, predicted_labels, dataset_type):
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1'])

    print(f"\n {dataset_type} Accuracy: {accuracy:.4f}")
    print(f"{dataset_type} Classification Report:\n{class_report}")
    print(f"{dataset_type} Confusion Matrix:\n{conf_matrix}")

    # --- Plot confusion matrix ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f"{dataset_type} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

# --- 8. Evaluate on both Train and Test sets ---
evaluate_classifier(y_train, train_predictions, "Train")
evaluate_classifier(y_test, test_predictions, "Test")
