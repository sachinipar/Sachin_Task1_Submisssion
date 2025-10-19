import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define dataset path
dataset_path = "dataset"
classes = os.listdir(dataset_path)

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()

data = []
labels = []

print("🔹 Loading dataset and extracting features...")

for label, cls in enumerate(classes):
    folder = os.path.join(dataset_path, cls)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        features = hog.compute(img)
        data.append(features)
        labels.append(label)

data = np.array(data).reshape(len(data), -1)
labels = np.array(labels)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM classifier
print("🔹 Training model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model_svm.pkl")
print("\n💾 Model saved as model_svm.pkl")
