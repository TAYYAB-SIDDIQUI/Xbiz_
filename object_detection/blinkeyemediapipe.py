import cv2
import mediapipe as mp
import math

# Eye aspect ratio calculation
def calculate_ear(upper, lower, left, right):
    vertical_dist = math.dist(upper, lower)
    horizontal_dist = math.dist(left, right)
    return vertical_dist / horizontal_dist

# Threshold for EAR
EAR_THRESHOLD = 0.30
CONSEC_FRAMES = 3

blink_counter = 0
total_blinks = 0

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get landmark coordinates and convert to pixel values
        left_eye = {
            "left": (int(landmarks[33].x * w), int(landmarks[33].y * h)),
            "right": (int(landmarks[133].x * w), int(landmarks[133].y * h)),
            "top": (int(landmarks[159].x * w), int(landmarks[159].y * h)),
            "bottom": (int(landmarks[145].x * w), int(landmarks[145].y * h))
        }

        right_eye = {
            "left": (int(landmarks[362].x * w), int(landmarks[362].y * h)),
            "right": (int(landmarks[263].x * w), int(landmarks[263].y * h)),
            "top": (int(landmarks[386].x * w), int(landmarks[386].y * h)),
            "bottom": (int(landmarks[374].x * w), int(landmarks[374].y * h))
        }

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye["top"], left_eye["bottom"], left_eye["left"], left_eye["right"])
        right_ear = calculate_ear(right_eye["top"], right_eye["bottom"], right_eye["left"], right_eye["right"])
        ear = (left_ear + right_ear) / 2.0

        # Blink detection logic
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                total_blinks += 1
                print(f"Blink detected! Total: {total_blinks}")
            blink_counter = 0

        # Draw on the frame
        cv2.putText(frame, f"Blinks: {total_blinks}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Optionally: Draw eye points
        for point in [33, 133, 159, 145, 362, 263, 386, 374]:
            cx, cy = int(landmarks[point].x * w), int(landmarks[point].y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    cv2.imshow("MediaPipe Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
