# detect_phone_and_face_webcam.py
import cv2
from ultralytics import YOLO
frame_count = 0
YOLO_SKIP_FRAMES = 2
device_boxes = []
# Load YOLOv8 model for object detection
model = YOLO("yolov8s.pt")
DEVICE_CLASS_NAMES = ["cell phone", "mobile phone", "phone", "laptop", "tv", "monitor"]
CONF_TH = 0.1

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO only every N frames
    if frame_count % YOLO_SKIP_FRAMES == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, imgsz=640, conf=CONF_TH, verbose=False)
        r = results[0]

        device_boxes = []  # Reset boxes

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name = model.model.names.get(cls_id, str(cls_id)).lower()


            if name in DEVICE_CLASS_NAMES:
                # Save device bounding box for fake face logic
                device_boxes.append((x1, y1, x2, y2))
                # Draw device box
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # =========== FACE DETECTION (OpenCV) ===========
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))

    for (fx, fy, fw, fh) in faces:
        face_box = (fx, fy, fx + fw, fy + fh)
        is_fake = False

        # Check if face is inside any device box
        for (dx1, dy1, dx2, dy2) in device_boxes:
            if (face_box[0] >= dx1 and face_box[1] >= dy1 and
                face_box[2] <= dx2 and face_box[3] <= dy2):
                is_fake = True
                break

        if is_fake:
            color = (0, 0, 255)  # Red for fake face
            label = "Fake Face"
        else:
            color = (255, 0, 0)  # Blue for real face
            label = "Face"

        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
        cv2.putText(frame, label, (fx, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # =========== Display ===========
    cv2.imshow("Device & Face Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
