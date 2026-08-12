from ultralytics import YOLO
import cv2
from voice.voice_manager import VoiceManager


# Load YOLO model
model = YOLO("yolov8n.pt")


# Minimum confidence required
CONFIDENCE_THRESHOLD = 0.60


# Number of consecutive frames required
REQUIRED_FRAMES = 3


# Number of frames an object can disappear
# before we consider it gone
MAX_MISSING_FRAMES = 10


# Create voice manager
voice = VoiceManager()


# Count consecutive frames for each tracking ID
track_counts = {}


# Track how many frames each ID has been missing
missing_counts = {}


# Track IDs that have already been announced
announced_tracks = set()


# Open webcam
camera = cv2.VideoCapture(1, cv2.CAP_MSMF)


while True:

    success, frame = camera.read()

    if not success:
        break


    # Run YOLO tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )


    boxes = results[0].boxes


    # IDs detected in the current frame
    current_track_ids = set()


    if boxes is not None:

        for box in boxes:

            # Get confidence
            confidence = float(box.conf[0])


            # Ignore low-confidence detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # Get object class
            class_id = int(box.cls[0])
            object_name = model.names[class_id]


            # Get tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])


            # Remember that this ID is currently visible
            current_track_ids.add(track_id)


            # Reset missing count
            missing_counts[track_id] = 0


            # Print detection
            print(
                f"{object_name} | "
                f"ID: {track_id} | "
                f"confidence: {confidence:.2f}"
            )


            # Count consecutive frames
            track_counts[track_id] = (
                track_counts.get(track_id, 0) + 1
            )


            # Announce only after the object has been
            # detected for enough consecutive frames
            if (
                track_counts[track_id] >= REQUIRED_FRAMES
                and track_id not in announced_tracks
            ):

                voice.speak(
                    f"{object_name} detected"
                )

                announced_tracks.add(track_id)


    # Handle objects that disappeared
    for track_id in list(track_counts.keys()):

        if track_id not in current_track_ids:

            missing_counts[track_id] = (
                missing_counts.get(track_id, 0) + 1
            )


            # Consider the object gone after
            # enough missing frames
            if (
                missing_counts[track_id]
                >= MAX_MISSING_FRAMES
            ):

                # Forget this object
                track_counts.pop(
                    track_id,
                    None
                )

                missing_counts.pop(
                    track_id,
                    None
                )

                announced_tracks.discard(
                    track_id
                )


    # Draw bounding boxes
    annotated_frame = results[0].plot()


    # Show camera
    cv2.imshow(
        "YOLO Object Tracking",
        annotated_frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
camera.release()
cv2.destroyAllWindows()