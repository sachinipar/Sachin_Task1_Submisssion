import cv2
import numpy as np
import joblib

hog = cv2.HOGDescriptor()
model = joblib.load("model_svm.pkl")

classes = ["R1_Real", "R1_Fake", "R2_Real", "R2_Fake"]

path = input("Enter path of image to test: ")

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
features = hog.compute(img).reshape(1, -1)

pred = model.predict(features)
prob = model.predict_proba(features)

print(f"\n🔍 Predicted Class: {classes[pred[0]]}")
print(f"Confidence: {np.max(prob)*100:.2f}%")

cv2.putText(img, classes[pred[0]], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
