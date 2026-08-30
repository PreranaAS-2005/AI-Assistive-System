import cv2
import os
import numpy as np

# ---------------------------------
# FACE DETECTOR
# ---------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ---------------------------------
# PREPROCESS FACE
# ---------------------------------

def preprocess_face(gray, x, y, w, h):
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (200, 200))

    # Normalize lighting
    face = cv2.equalizeHist(face)

    return face


# ---------------------------------
# LOAD KNOWN FACES
# ---------------------------------

known_faces_path = "known_faces"

faces_data = []
labels = []
label_names = {}

current_label = 0

for person_name in os.listdir(known_faces_path):

    person_path = os.path.join(
        known_faces_path,
        person_name
    )

    if not os.path.isdir(person_path):
        continue

    print(f"\nLoading images for: {person_name}")

    images_loaded = 0

    for filename in os.listdir(person_path):

        image_path = os.path.join(
            person_path,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Could not read: {filename}")
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(detected_faces) > 0:

            # Select largest detected face
            x, y, w, h = max(
                detected_faces,
                key=lambda rect: rect[2] * rect[3]
            )

            face = preprocess_face(
                gray, x, y, w, h
            )

            faces_data.append(face)
            labels.append(current_label)

            images_loaded += 1

        else:
            print(f"⚠️ No face detected in: {filename}")

    if images_loaded > 0:

        label_names[current_label] = person_name

        print(f"✅ Loaded {images_loaded} image(s)")

        current_label += 1


# ---------------------------------
# CHECK TRAINING DATA
# ---------------------------------

if len(faces_data) == 0:
    print("❌ No training faces found.")
    exit()

print("\n------------------------------")
print("KNOWN PEOPLE")

for label, name in label_names.items():
    print(f"{label}: {name}")

print("------------------------------")
print(f"Total training images: {len(faces_data)}")


# ---------------------------------
# TRAIN RECOGNIZER
# ---------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces_data,
    np.array(labels)
)

print("\n✅ Face recognizer trained successfully!")


# ---------------------------------
# START CAMERA
# ---------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera")
    exit()

print("✅ Camera started. Press 'q' to quit.")


# ---------------------------------
# RECOGNITION SETTINGS
# ---------------------------------

CONFIDENCE_THRESHOLD = 60


# ---------------------------------
# LIVE FACE RECOGNITION
# ---------------------------------

while True:

    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in detected_faces:

        face = preprocess_face(
            gray, x, y, w, h
        )

        # Predict
        label, confidence = recognizer.predict(face)

        predicted_name = label_names.get(
            label,
            "Unknown"
        )

        # Lower confidence = better match
        if confidence > CONFIDENCE_THRESHOLD:
            predicted_name = "Unknown"

        # Print raw prediction for debugging
        print(
            f"Prediction: {predicted_name} | "
            f"Confidence: {confidence:.2f}"
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display result
        cv2.putText(
            frame,
            f"{predicted_name} ({confidence:.1f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Phase 2 - Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()