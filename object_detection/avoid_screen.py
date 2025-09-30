import cv2
  # import from your module
import cv2

def detect_rectangles(frame, min_area=16):
    """
    Detects large rectangular contours in the frame that might correspond to screens.

    Args:
        frame: BGR image frame (numpy array)
        min_area: minimum area of rectangle to consider (default 5000)

    Returns:
        List of bounding rectangles as (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 1, 18)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > min_area:
            rectangles.append(cv2.boundingRect(approx))

    return rectangles
# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3, minSize=(8, 8))

    # Detect potential screens (rectangles)
    screens = detect_rectangles(frame)

    for (x, y, w, h) in faces:
        face_rect = (x, y, w, h)

        # Check if face is inside any detected screen rectangle
        inside_screen = False
        for (rx, ry, rw, rh) in screens:
            if (x > rx and y > ry and x + w < rx + rw and y + h < ry + rh):
                inside_screen = True
                break

        if inside_screen:
            # Skip this face, likely face on a screen
            cv2.putText(frame, "Face on screen ignored", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue

        # Else draw rectangle and process face normally
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Real face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection with Screen Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
