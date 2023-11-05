import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model from the saved model directory
# emotion_model = load_model('/Volumes/X/git/ece327/Emotion_detection_with_CNN-main/model/emotion_model.h5')
emotion_model = load_model('/Volumes/X/git/ece327/Emotion_detection_with_CNN-main/model/emotion_model')
print("Loaded model from disk")

# Start the webcam feed or video file
cap = cv2.VideoCapture(0)
# Video path
#cap = cv2.VideoCapture("/Users/sjsb/git/ece327/Emotion_detection_with_CNN-main/emotion_sample_1.mp4")

while True:
    # Read frame from the camera or video file
    ret, frame = cap.read()

    # If frame is read correctly ret is True. If not, break from the loop.
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    try:
        # Resize the frame for consistent processing
        frame = cv2.resize(frame, (1280, 720))
    except cv2.error as e:
        print(f"Error resizing frame: {e}")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the haar cascade classifier to detect faces
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)

        # Get region of interest (ROI)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        cropped_img = cropped_img / 255.0  # Normalize pixel values

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]
        cv2.putText(frame, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
