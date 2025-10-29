# toothpaste_template_matching.py
# Uses multiple templates (different angles/brands) + multi-scale matching

import cv2
import numpy as np
import os

TEMPLATE_DIR = "templates"
THRESHOLD = 0.7

templates = []
for tmp_file in os.listdir(TEMPLATE_DIR):
    if tmp_file.endswith(('.png', '.jpg')):
        tmp = cv2.imread(os.path.join(TEMPLATE_DIR, tmp_file), cv2.IMREAD_GRAYSCALE)
        templates.append(tmp)

def detect_template(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for template in templates:
        h, w = template.shape
        for scale in np.linspace(0.5, 1.5, 20):
            resized = cv2.resize(template, (int(w*scale), int(h*scale)))
            if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= THRESHOLD)
            for pt in zip(*loc[::-1]):
                x2 = pt[0] + int(w*scale)
                y2 = pt[1] + int(h*scale)
                cv2.rectangle(frame, pt, (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Toothpaste", (pt[0], pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

# Create 'templates/' folder and put 3-5 cropped toothpaste images
if __name__ == "__main__":
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR)
        print(f"Put cropped toothpaste images in '{TEMPLATE_DIR}/' folder!")
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = detect_template(frame)
        cv2.imshow("Template Matching Toothpaste", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()