import cv2
import face_recognition
import pickle
import time

def capture_image():
    cap = cv2.VideoCapture(0)  # 0 for default camera
    time.sleep(2)  # Wait for 2 seconds to let the camera initialize
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("user_face.jpg", frame)
    else:
        print("Error: Could not capture an image.")
    cap.release()
    cv2.destroyAllWindows()

def generate_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        return face_encodings[0]
    else:
        print("No faces found in the image.")
        return None

def save_face_encoding(face_encoding, filename="face_encoding.pkl"):
    if face_encoding is not None:
        with open(filename, "wb") as f:
            pickle.dump(face_encoding, f)
    else:
        print("Face encoding not saved as it is None.")

def initial_configuration():
    capture_image()
    face_encoding = generate_face_encoding("user_face.jpg")
    save_face_encoding(face_encoding)

if __name__ == "__main__":
    initial_configuration()
