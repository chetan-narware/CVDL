# 🚀 Image Feature Extraction & Classification

## 📌 Overview

This project explores various image processing techniques to extract meaningful features for classification. The extracted features can be used to classify images into different categories (e.g., **Dog, Cat, Horse, Digits from MNIST**).

## 🖼️ Features Extracted

### 🔹 Edge Detection

- **Sobel Filter** (Detects vertical and horizontal edges)
- **Canny Edge Detection** (Multi-stage edge detection algorithm)
- **Laplacian Filter** (Finds edges using the second derivative)

### 🔹 Corner Detection

- **Harris Corner Detector** (Identifies corner points in an image)

### 🔹 Keypoint Detection

- **SIFT (Scale-Invariant Feature Transform)** (Detects keypoints and descriptors for matching)
- **ORB (Alternative to SIFT)** (For systems where SIFT is unavailable)

### 🔹 Texture Analysis

- **Gabor Filter** (Extracts texture information)

### 🔹 Feature Extraction for Classification

- **Histogram of Oriented Gradients (HOG)** (Extracts shape-based features)
- **Feature Extraction & Classification on MNIST Dataset** (Using HOG + ANN for digit classification)

## 🛠️ Installation

Ensure you have Python and the required libraries installed.

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib scikit-image tensorflow
```

## 📜 Usage

### 🔹 Apply Edge Detection

```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
canny_edges = cv2.Canny(image, 100, 200)

plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.show()
```

### 🔹 Apply SIFT (Scale-Invariant Feature Transform)

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
image_sift = cv2.drawKeypoints(image, keypoints, None)
plt.imshow(image_sift, cmap="gray")
plt.title("SIFT Keypoints")
plt.show()
```

### 🔹 Apply HOG for Feature Extraction

```python
from skimage.feature import hog
hog_features, hog_image = hog(image, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG Features")
plt.show()
```

### 🔹 Feature Extraction & Classification on MNIST using ANN

```python
import tensorflow as tf
from tensorflow import keras
from skimage.feature import hog
import numpy as np

# Load MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and extract HOG features
def extract_hog_features(images):
    return np.array([hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True) for img in images])

X_train_hog = extract_hog_features(X_train) / 255.0
X_test_hog = extract_hog_features(X_test) / 255.0

# Define ANN model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_hog.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_hog, y_train, epochs=20, batch_size=128, validation_split=0.1)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test_hog, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

📢 **Contributions & Feedback** are welcome! 🚀
