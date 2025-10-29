# toothpaste_hog_svm.py
# Uses HOG features + pre-trained SVM to detect toothpaste tubes
# Requires a small training set (you can create it with labelImg or similar)

import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog

# ------------------- 1. Train a simple SVM (run once) -------------------
def train_svm(pos_dir="train/pos", neg_dir="train/neg", model_path="toothpaste_svm.pkl"):
    hog_features = []
    labels = []

    # Positive samples (cropped toothpaste tubes)
    for img_file in os.listdir(pos_dir):
        img = cv2.imread(os.path.join(pos_dir, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))
        fd = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(fd)
        labels.append(1)

    # Negative samples
    for img_file in os.listdir(neg_dir):
        img = cv2.imread(os.path.join(neg_dir, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))
        fd = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(fd)
        labels.append(0)

    X = np.array(hog_features)
    y = np.array(labels)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    joblib.dump(svm, model_path)
    print(f"Model saved to {model_path}")

# ------------------- 2. Sliding-window detector -------------------
def detect_with_svm(frame, model_path="toothpaste_svm.pkl", win_size=(64, 128), step=32):
    if not os.path.exists(model_path):
        print("Train the model first!")
        return frame

    svm = joblib.load(model_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = []
    for y in range(0, gray.shape[0] - win_size[1], step):
        for x in range(0, gray.shape[1] - win_size[0], step):
            patch = gray[y:y+win_size[1], x:x+win_size[0]]
            patch = cv2.resize(patch, win_size)
            fd = hog(patch, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
            fd = fd.reshape(1, -1)
            _, result = svm.predict(fd)
            if result[0][0] == 1:
                detections.append((x, y, x + win_size[0], y + win_size[1]))

    # Non-max suppression (simple)
    boxes = cv2.groupRectangles(detections, groupThreshold=1, eps=0.2)[0] if detections else []
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Toothpaste", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

# ------------------- 3. Demo -------------------
if __name__ == "__main__":
    # Uncomment to train (you need pos/ and neg/ folders)
    # train_svm()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = detect_with_svm(frame)
        cv2.imshow("HOG + SVM Toothpaste Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()