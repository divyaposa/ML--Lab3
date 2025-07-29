import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# --- 1. Load dataset and flatten images ---
image_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.view(-1))  # flatten image to vector
])

image_dataset = datasets.ImageFolder(r'C:\Users\<your-username>\Desktop\Dataset', transform=image_transform)



# --- 2. Select two image feature vectors ---
image_vector_1 = image_dataset[0][0].numpy()
image_vector_2 = image_dataset[1][0].numpy()

# --- 3. Calculate Minkowski distances (r = 1 to 10) ---
minkowski_r_values = list(range(1, 11))
minkowski_distances = []

for r in minkowski_r_values:
    distance = np.sum(np.abs(image_vector_1 - image_vector_2) ** r) ** (1 / r)
    minkowski_distances.append(distance)

# --- 4. Plot distance vs r ---
plt.plot(minkowski_r_values, minkowski_distances, marker='o')
plt.title('Minkowski Distance Between Two Image Vectors (r = 1 to 10)')
plt.xlabel('r (Minkowski Power Parameter)')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# --- 5. Print distances ---
for r, distance in zip(minkowski_r_values, minkowski_distances):
    print(f"r = {r}: Distance = {distance:.4f}")
