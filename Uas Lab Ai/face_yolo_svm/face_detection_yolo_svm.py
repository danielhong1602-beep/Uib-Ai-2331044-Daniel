import cv2
import joblib
import numpy as np
from ultralytics import YOLO
from skimage.feature import hog

# ===== LOAD MODELS =====
face_model = YOLO(r'C:\Users\ASUS\Lab_Ai\Uas Lab Ai\face_yolo_svm\models\yolo\yolov8n-face-lindevs.pt')
svm = joblib.load(r'C:\Users\ASUS\Lab_Ai\Uas Lab Ai\face_yolo_svm\models\svm\training_svm.pkl')

CLASSES = ["Angry", "Happy", "Neutral", "Sad"]

COLOR_MAP = {
    "Angry": (0, 0, 255),     # merah
    "Happy": (0, 255, 0),     # hijau
    "Neutral": (255, 255, 0), # kuning
    "Sad": (255, 0, 0)        # biru
}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ===== WEBCAM =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame, conf=0.5)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))

            features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            scores = svm.decision_function([features])[0]
            probs = softmax(scores)

            pred = np.argmax(probs)
            emotion = CLASSES[pred]
            percent = probs[pred] * 100

            color = COLOR_MAP[emotion]
            label = f"{emotion}: {percent:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
