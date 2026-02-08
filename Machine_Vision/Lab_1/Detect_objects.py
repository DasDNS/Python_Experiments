import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLO model
# After training, the best model is saved in 'runs/classify/yolo_emotion_model/weights/best.pt'
# You may need to update this path depending on where the train script saved it.
try:
    model = YOLO('runs/classify/yolo_emotion_model/weights/best.pt')
except:
    print("Could not find trained model. Loading default for demonstration.")
    model = YOLO('yolov8n-cls.pt')

video = cv2.VideoCapture(0)

# We still use Haar Cascade to find the face first
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascade_path)

while True:
    ret, frame = video.read()
    if not ret:
        break
        
    # Convert to grayscale for Face Detection (Haar Cascade needs gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        # 1. Crop the face
        # NOTE: YOLO expects color images, so we crop from the original 'frame', not 'gray'
        sub_face_img = frame[y:y+h, x:x+w]
        
        # 2. Run Inference using YOLO
        # YOLO handles resizing and normalization internally
        results = model(sub_face_img, verbose=False)
        
        # 3. Get the class with the highest probability
        # results[0].probs.top1 gets the index of the top prediction
        top1_index = results[0].probs.top1
        class_name = results[0].names[top1_index]
        confidence = results[0].probs.top1conf.item()
        
        # Prepare label text
        label_text = f"{class_name} ({confidence:.2f})"
        
        # 4. Draw rectangles and text (same style as your original code)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()