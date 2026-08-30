import cv2
import os
import numpy as np
import pyttsx3
import threading
from deepface import DeepFace
from collections import deque, Counter


# ---------------------------------
# SETTINGS
# ---------------------------------

KNOWN_FACES_PATH = "known_faces"
MODEL_NAME = "Facenet512"
DISTANCE_THRESHOLD = 0.35


# ---------------------------------
# PREDICTION STABILITY SETTINGS
# ---------------------------------

PREDICTION_HISTORY_SIZE = 5
MIN_STABLE_COUNT = 3

prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)


# ---------------------------------
# VOICE SETTINGS
# ---------------------------------

speech_lock = threading.Lock()


def speak(text):
    def run_speech():
        try:
            with speech_lock:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()

        except Exception as e:
            print(f"⚠️ Speech error: {e}")

    speech_thread = threading.Thread(
        target=run_speech,
        daemon=True
    )

    speech_thread.start()


last_announced_name = None


# ---------------------------------
# STORE KNOWN FACE DATA
# ---------------------------------

known_embeddings = []
known_names = []


# ---------------------------------
# LOAD KNOWN FACE EMBEDDINGS
# ---------------------------------

print("Loading known faces...\n")

for person_name in os.listdir(KNOWN_FACES_PATH):

    person_path = os.path.join(
        KNOWN_FACES_PATH,
        person_name
    )

    if not os.path.isdir(person_path):
        continue

    print(f"Processing: {person_name}")

    images_loaded = 0

    for filename in os.listdir(person_path):

        image_path = os.path.join(
            person_path,
            filename
        )

        try:

            embedding_objects = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False
            )

            embedding = embedding_objects[0]["embedding"]

            known_embeddings.append(
                np.array(embedding)
            )

            known_names.append(person_name)

            images_loaded += 1

        except Exception as e:

            print(
                f"⚠️ Could not process {filename}"
            )

            print(f"Error: {e}")

    print(
        f"✅ Loaded {images_loaded} image(s)\n"
    )


# ---------------------------------
# CHECK LOADED FACES
# ---------------------------------

if len(known_embeddings) == 0:

    print("❌ No known faces loaded.")
    exit()


print("------------------------------")
print("KNOWN FACE EMBEDDINGS LOADED")
print("------------------------------")

for name in sorted(set(known_names)):

    count = known_names.count(name)

    print(f"{name}: {count} image(s)")


known_embeddings = np.array(known_embeddings)


# ---------------------------------
# COSINE DISTANCE
# ---------------------------------

def cosine_distance(a, b):

    return 1 - np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )


# ---------------------------------
# START CAMERA
# ---------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open camera")
    exit()


print("\n✅ Camera started.")
print("Press 'q' to quit.")


# ---------------------------------
# LIVE FACE RECOGNITION
# ---------------------------------

while True:

    success, frame = camera.read()

    if not success:

        print("❌ Could not read camera frame")
        break


    # Detect faces
    try:

        detected_faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )

    except Exception as e:

        print(
            f"⚠️ Face detection error: {e}"
        )

        detected_faces = []


    # Reset when no face is visible
    if len(detected_faces) == 0:

        prediction_history.clear()

        last_announced_name = None


    # Process each face
    for detected_face in detected_faces:

        try:

            face = detected_face["face"]

            facial_area = detected_face[
                "facial_area"
            ]

            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]


            # Convert to uint8
            face = (
                face * 255
            ).astype(np.uint8)


            # Create embedding
            result = DeepFace.represent(
                img_path=face,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False
            )

            live_embedding = np.array(
                result[0]["embedding"]
            )


            # Calculate distances
            distances = []

            for known_embedding in known_embeddings:

                distance = cosine_distance(
                    live_embedding,
                    known_embedding
                )

                distances.append(distance)


            # Best match
            best_index = np.argmin(distances)

            best_distance = distances[
                best_index
            ]


            # Raw prediction
            if best_distance < DISTANCE_THRESHOLD:

                raw_prediction = known_names[
                    best_index
                ]

            else:

                raw_prediction = "Unknown"


            # Prediction stability
            prediction_history.append(
                raw_prediction
            )

            prediction_counts = Counter(
                prediction_history
            )

            most_common_name, count = (
                prediction_counts
                .most_common(1)[0]
            )


            if count >= MIN_STABLE_COUNT:

                predicted_name = most_common_name

            else:

                predicted_name = "Unknown"


            # ---------------------------------
            # VOICE ANNOUNCEMENT
            # ---------------------------------

            if (
                predicted_name != "Unknown"
                and predicted_name != last_announced_name
            ):

                announcement = (
                    f"{predicted_name} detected"
                )

                print(
                    f"\n🔊 SPEAKING: {announcement}\n"
                )

                speak(announcement)

                last_announced_name = (
                    predicted_name
                )


            # Show result in terminal
            print(
                f"Raw: {raw_prediction} | "
                f"Stable: {predicted_name} | "
                f"Distance: {best_distance:.3f}"
            )


            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # Display prediction
            cv2.putText(
                frame,
                f"{predicted_name} ({best_distance:.3f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        except Exception as e:

            print(
                "⚠️ Could not recognize face"
            )

            print(f"Error: {e}")


    # Show camera
    cv2.imshow(
        "Phase 2 - DeepFace Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ---------------------------------
# CLEANUP
# ---------------------------------

camera.release()

cv2.destroyAllWindows()