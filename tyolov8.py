# toothpaste_yolov8.py
# Uses Ultralytics YOLOv8 - state-of-the-art object detection
# Install: pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

# Load a custom-trained YOLOv8 model (or use pre-trained + fine-tune)
# Download a sample trained model or train your own: https://docs.ultralytics.com/
model = YOLO("toothpaste_yolov8n.pt")  # <-- your trained model

def detect_yolo(frame):
    results = model(frame, conf=0.4)[0]  # confidence threshold
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        if model.names[cls] == "toothpaste":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Toothpaste {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

# Webcam demo
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = detect_yolo(frame)
        cv2.imshow("YOLOv8 Toothpaste Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()