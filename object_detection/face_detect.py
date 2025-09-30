import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

prev_face_position = None  # Store previous face center

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(8, 8))

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get center of face rectangle
        current_face_position = (x + w // 2, y + h // 2)

        if prev_face_position is not None:
            dx = current_face_position[0] - prev_face_position[0]
            dy = current_face_position[1] - prev_face_position[1]

            threshold = 10  # Minimum movement to consider

            if abs(dx) < threshold and abs(dy) < threshold:
                movement = "Still"
            else:
                if abs(dx) > abs(dy):
                    if dx > 0:
                        movement = "Moved Right"
                    else:
                        movement = "Moved Left"
                else:
                    if dy > 0:
                        movement = "Moved Down"
                    else:
                        movement = "Moved Up"

            # Show movement direction
            cv2.putText(frame, f"Movement: {movement}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update previous face position
        prev_face_position = current_face_position

        # Only track one face
        break

    cv2.imshow('Face Movement Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
