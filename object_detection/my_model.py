import cv2
import joblib
model=joblib.load("tayyab_fake_real_face (1).pkl")
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

prev_face_position = None  # Store previous face center

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    if not ret:
        break
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(32,32))
    img_array = np.expand_dims(img, axis=0)
    pr=model.predict(img_array)
    if pr[0][0]>0.5:
        ans="fake"
    else:
        ans="real"
    cv2.putText(frame, f"predict: {ans}", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow('Face Movement Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()