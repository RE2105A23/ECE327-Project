<<<<<<< HEAD
=======
"""
>>>>>>> e85ce94 (Initial commit)
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    cv2.imshow('Webcam Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
<<<<<<< HEAD
=======
"""

import cv2
import time

# Initialize the camera
cap = cv2.VideoCapture(0)

# Wait for camera to initialize
time.sleep(2)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Capture a single frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
    else:
        print("Error: Could not capture an image.")

    # Release the camera
    cap.release()

>>>>>>> e85ce94 (Initial commit)
