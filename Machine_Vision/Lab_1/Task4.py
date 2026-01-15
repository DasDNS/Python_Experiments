# ==========================================
# TASK 4: YOLO Face Detection + CNN Emotion
# ==========================================

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -------------------------------
# 1. LOAD MODELS
# -------------------------------

# Load YOLOv8 face detection model (pretrained)
yolo_model = YOLO("yolov8n.pt")   # lightweight YOLO model

# Load trained CNN emotion model
emotion_model = load_model("facial_expression_cnn_model.h5")

# Emotion labels (same order as training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------------------------------
# 2. START WEBCAM
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("✅ YOLO + CNN Emotion Detection Started")

# -------------------------------
# 3. REAL-TIME DETECTION LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO face detection
    results = yolo_model(frame, conf=0.4)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop detected face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # -------------------------------
            # 4. PREPROCESS FACE FOR CNN
            # -------------------------------
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

            # -------------------------------
            # 5. EMOTION PREDICTION
            # -------------------------------
            prediction = emotion_model.predict(reshaped_face, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]

            # -------------------------------
            # 6. DISPLAY RESULTS
            # -------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    cv2.imshow("YOLO + CNN Facial Expression Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 7. CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Program terminated")
