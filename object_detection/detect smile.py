import cv2
import mediapipe as mp
import math

# Calculate Euclidean distance between two points

def euclidean_dist(p1, p2):
    return math.dist(p1, p2)

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    # Extract eye landmarks
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    
    # Vertical distances
    A = euclidean_dist(points[1], points[5])
    B = euclidean_dist(points[2], points[4])
    # Horizontal distance
    C = euclidean_dist(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(landmarks, mouth_indices, img_w, img_h):
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in mouth_indices]

    # Vertical distances between lip landmarks
    A = euclidean_dist(points[13], points[19])  # 14 and 20 in 1-based index (top inner lip, bottom inner lip)
    B = euclidean_dist(points[14], points[18])  # 15 and 19 (top outer lip, bottom outer lip)
    # Horizontal distance between mouth corners
    C = euclidean_dist(points[0], points[6])  # 49 and 55 (left and right mouth corners)

    mar = (A + B) / (2.0 * C)
    return mar

# MediaPipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Eye and mouth landmark indices (from MediaPipe face mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 178, 87, 14, 317, 402]

# Thresholds (tweak experimentally)
EAR_THRESHOLD = 0.23  # Below this = eyes squinting
MAR_THRESHOLD = 0.4   # Above this = smiling

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR (mouth aspect ratio)
        mar = mouth_aspect_ratio(landmarks, MOUTH_IDX, img_w, img_h)

        # Detect smile and eye squint
        smiling = mar > MAR_THRESHOLD
        eyes_squint = ear < EAR_THRESHOLD

        if smiling:
            if eyes_squint:
                smile_type = "Real Smile "
            else:
                smile_type = "Fake Smile "
        else:
            smile_type = "No Smile"

        # Draw text
        cv2.putText(frame, f"Smile: {smile_type}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optional: draw landmarks for debugging
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX + MOUTH_IDX:
            x, y = int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Real vs Fake Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
