import cv2

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from default webcam
cap = cv2.VideoCapture(0)

prev_face_position = None  # To store previous face position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (required for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_x,all_y=gray.shape

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,minSize=(8,8))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Current face position (center of the face rectangle)
        current_face_position = (all_x + w//2, all_y + h//2)

        if prev_face_position is not None:
            dx = current_face_position[0] - prev_face_position[0]
            dy = current_face_position[1] - prev_face_position[1]

            # Calculate movement distance
            movement = (dx, dy)
            if dx>0 and dy>0:
                move="moved towards camera"
            elif dx==0 and dy==0:
                move="Still position"
            else:
                move="moved far from camera"
            movement_text = f"Movement: {move}"

            # Display movement on the frame
            cv2.putText(frame, movement_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update previous face position for next frame
        prev_face_position = current_face_position

        # Only track the first detected face for simplicity
        break

    cv2.imshow('Face Movement Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
