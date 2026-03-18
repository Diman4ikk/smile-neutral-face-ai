import cv2
import torch
import torch.nn as nn
from model import res_net_model
import mediapipe as mp
from collections import deque
import statistics

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
model=res_net_model()
emotion_history = deque(maxlen=5)
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

#face_cascade = cv2.CascadeClassifier(
  #  cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#)

cap = cv2.VideoCapture(0)

class_name = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Neutral", 
    4: "Sad", 
    5: "Smile", 
    6: "Surprise"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key= cv2.waitKey(1) & 0xFF
    frame = cv2.flip(frame, 1)
    ih, iw, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x=int(bbox.xmin*iw)
            y=int(bbox.ymin*ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            x, y = max(0, x), max(0, y)
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            try:
                
                resized=cv2.resize(face,(64,64))
                gray_face = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                rgb_gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB)
                tensor=torch.from_numpy(rgb_gray_face).float()/255.0
                tensor=(tensor-0.5)/0.5
                tensor = tensor.permute(2, 0, 1)
                tensor = tensor.unsqueeze(0)

                with torch.inference_mode():
                    prediction = model(tensor)
                    probabilities = torch.softmax(prediction, dim=1)
                    conf_tensor, pred_class_tensor = torch.max(probabilities, dim=1)
                    
                    confidence = conf_tensor.item() * 100
                    predicted_class = pred_class_tensor.item()
                emotion_history.append(predicted_class)

                smoothed_class = statistics.mode(emotion_history)
                label_text = f"{class_name[predicted_class]} ({confidence:.1f}%)"
                colors = {0: (0,0,255), 3: (0,255,0), 6: (255,0,0)} # Можно расширить
                color = colors.get(predicted_class, (255, 255, 255))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)                             
            except Exception as e:
                print(f"Ошибка обработки лица: {e}")
                continue

    cv2.imshow("Camera", frame)
  

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
