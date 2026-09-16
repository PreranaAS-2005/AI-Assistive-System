import cv2
import os
import sys
import numpy as np
import threading
from collections import deque, Counter
from deepface import DeepFace
from voice.voice_manager import VoiceManager


# ==========================================
# SETTINGS
# ==========================================

KNOWN_FACES_PATH = "known_faces"
MODEL_NAME = "Facenet512"
DISTANCE_THRESHOLD = 0.35

PREDICTION_HISTORY_SIZE = 5
MIN_STABLE_COUNT = 3
FACE_RECOGNITION_INTERVAL = 5

CAMERA_INDEX = 1


# ==========================================
# INITIALIZATION
# ==========================================

voice = VoiceManager()
known_embeddings = []
known_names = []


def load_known_faces():
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []

    print("\n[INFO] Loading known faces...")
    if not os.path.exists(KNOWN_FACES_PATH):
        print(f"[ERROR] Folder not found: {KNOWN_FACES_PATH}")
        return

    for person_name in os.listdir(KNOWN_FACES_PATH):
        person_path = os.path.join(KNOWN_FACES_PATH, person_name)
        if not os.path.isdir(person_path):
            continue

        images_loaded = 0
        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            try:
                embedding_objects = DeepFace.represent(
                    img_path=image_path,
                    model_name=MODEL_NAME,
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                embedding = embedding_objects[0]["embedding"]
                known_embeddings.append(np.array(embedding))
                known_names.append(person_name)
                images_loaded += 1
            except Exception as e:
                print(f"⚠️ Could not process {filename}: {e}")

        print(f"✅ Loaded {images_loaded} image(s) for {person_name}")

    print(f"👤 Total known face images: {len(known_embeddings)}\n")


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


load_known_faces()

if len(known_embeddings) == 0:
    print("❌ No known faces available.")
    sys.exit(1)

# Open DroidCam with CAP_MSMF
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
if not camera.isOpened():
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not camera.isOpened():
    print("❌ Could not open camera.")
    sys.exit(1)

print("[INFO] Face Recognition active. Press 'q' to return to controller.")

prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
last_announced_name = None
frame_count = 0

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to grab frame.")
            break

        frame_count += 1

        if frame_count % FACE_RECOGNITION_INTERVAL == 0:
            try:
                detected_faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="opencv",
                    enforce_detection=False,
                )
            except Exception:
                detected_faces = []

            if len(detected_faces) == 0:
                prediction_history.clear()
                last_announced_name = None

            for detected_face in detected_faces:
                try:
                    face = detected_face["face"]
                    facial_area = detected_face["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

                    face_uint8 = (face * 255).astype(np.uint8)

                    result = DeepFace.represent(
                        img_path=face_uint8,
                        model_name=MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                    live_embedding = np.array(result[0]["embedding"])

                    distances = [cosine_distance(live_embedding, ke) for ke in known_embeddings]
                    best_index = np.argmin(distances)
                    best_distance = distances[best_index]

                    raw_prediction = known_names[best_index] if best_distance < DISTANCE_THRESHOLD else "Unknown"

                    prediction_history.append(raw_prediction)
                    counts = Counter(prediction_history)
                    most_common, count = counts.most_common(1)[0]
                    stable_name = most_common if count >= MIN_STABLE_COUNT else "Unknown"

                    if stable_name != "Unknown" and stable_name != last_announced_name:
                        announcement = f"{stable_name} detected"
                        print(f"🔊 {announcement}")
                        threading.Thread(target=voice.speak, args=(announcement,), daemon=True).start()
                        last_announced_name = stable_name

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{stable_name} ({best_distance:.2f})",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                except Exception as e:
                    pass

        cv2.imshow("Face Recognition Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()