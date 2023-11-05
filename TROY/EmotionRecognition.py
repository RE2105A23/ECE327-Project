import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the model with weights included
model = load_model('/Users/sjsb/git/ece327/datasets/emotion_model.h5')

# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = cv2.resize(image, (48, 48))  # Resize image to expected input shape
    feature = np.array(feature).reshape(1, 48, 48, 1)  # Reshape for the model
    return feature / 255.0  # Normalize pixel values


webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, frame = webcam.read()
    if not ret:
        break  # If the frame is not properly captured, exit the loop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = extract_features(roi_gray)
        prediction = model.predict(roi_gray)
        emotion = labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'Esc' is pressed
        break

webcam.release()
cv2.destroyAllWindows()