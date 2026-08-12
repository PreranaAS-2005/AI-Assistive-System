from ultralytics import YOLO
import cv2
from voice.voice_manager import VoiceManager


# Load YOLO model
model = YOLO("yolov8n.pt")

# Create voice manager
voice = VoiceManager()

# Objects detected in the previous frame
previous_objects = set()

# Open webcam
camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        break

    # Run YOLO detection
    results = model(frame)

    boxes = results[0].boxes

    # Objects detected in the current frame
    current_objects = set()

    for box in boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        print(f"{object_name} : {confidence:.2f}")

        current_objects.add(object_name)

    # Find objects that appeared now
    new_objects = current_objects - previous_objects

    # Announce new objects
    for object_name in new_objects:

        voice.speak(f"{object_name} detected")

    # Current frame becomes previous frame
    previous_objects = current_objects

    # Display detections
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Voice Manager", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()