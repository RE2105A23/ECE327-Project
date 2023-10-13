import cv2
import face_recognition
import pickle

def capture_image():
    cap = cv2.VideoCapture(0)  # 0 for default camera
    ret, frame = cap.read()
    cv2.imwrite("user_face.jpg", frame)
    cap.release()
    cv2.destroyAllWindows()

def generate_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    return face_encoding

def save_face_encoding(face_encoding, filename="face_encoding.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(face_encoding, f)

def initial_configuration():
    capture_image()
    face_encoding = generate_face_encoding("user_face.jpg")
    save_face_encoding(face_encoding)

if __name__ == "__main__":
    initial_configuration()
