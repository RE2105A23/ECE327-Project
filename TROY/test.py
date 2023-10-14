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
