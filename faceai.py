import cv2
import torch
import torch.nn as nn
import os.path
from model import SimpleCNN

smile_folder = r"C:\Users\Dima\faceai\data\smile"
neutral_folder = r"C:\Users\Dima\faceai\data\neutral"

smile_count=0
neutral_count=0

model = SimpleCNN()
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

class_name = {0: "Neutral", 1: "Smile"}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key= cv2.waitKey(1) & 0xFF

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces_resized = cv2.resize(gray_face, (64, 64))

            tensor = torch.from_numpy(faces_resized).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                prediction = model(tensor)
                predicted_class = torch.argmax(prediction, dim=1).item()

            label_text = class_name[predicted_class]

            color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)
            if key==ord('s'):
                smile_count+=1
                file_name=f"smile_{smile_count}.jpg"
                full_path = os.path.join(smile_folder, file_name)
                cv2.imwrite(full_path,faces_resized)
                print(f"Сохранено в smile: {full_path}")
            if key==ord('n'):
                neutral_count+=1
                file_name=f"netural_{neutral_count}.jpg"
                full_path = os.path.join(neutral_folder, file_name)
                cv2.imwrite(full_path,faces_resized)
                print(f"Сохранено в netural: {full_path}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        except Exception as e:
            print(f"Ошибка обработки лица: {e}")
            continue

    cv2.imshow("Camera", frame)
  

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
