"""
Live camera test for the DetectionFilter.

This is a temporary diagnostic program.
It does NOT modify or run navigation_main.py.

Run:
    python detection_filter_camera_test.py

The terminal prints only when the STABLE detection set changes,
instead of printing every camera frame.
"""

from ultralytics import YOLO
import cv2

from navigation.spatial_analyzer import analyze_detection
from navigation.detection_filter import DetectionFilter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX = 2
CONFIDENCE_THRESHOLD = 0.60
IMAGE_SIZE = 320


# ---------------------------------------------------------------------------
# Load YOLO
# ---------------------------------------------------------------------------

print("[INFO] Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("[INFO] YOLO model loaded.")


# ---------------------------------------------------------------------------
# Open camera
# ---------------------------------------------------------------------------

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("[ERROR] Camera could not be opened.")
    raise SystemExit(1)

print("[INFO] Camera connected.")


# ---------------------------------------------------------------------------
# Detection filter
# ---------------------------------------------------------------------------

detector_filter = DetectionFilter(
    min_confidence=CONFIDENCE_THRESHOLD,
    stability_frames=3,
    missing_frames_to_remove=5,
)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

print()
print("[INFO] Detection Filter Camera Test")
print("-----------------------------------")
print("Show ONE object at a time.")
print()
print("Test:")
print("  1. person")
print("  2. laptop")
print("  3. TV")
print("  4. chair")
print("  5. bottle")
print("  6. phone")
print()
print("Keep each object visible for 5-10 seconds.")
print("Move the camera away before showing the next object.")
print("Press Q to quit.")
print()


# ---------------------------------------------------------------------------
# Remember the last stable result.
# This prevents terminal spam.
# ---------------------------------------------------------------------------

last_stable = None


try:

    while True:

        # ---------------------------------------------------------------
        # Read camera frame
        # ---------------------------------------------------------------

        success, frame = camera.read()

        if not success:
            print("[ERROR] Failed to read camera frame.")
            break

        frame_height, frame_width = frame.shape[:2]


        # ---------------------------------------------------------------
        # YOLO detection
        # ---------------------------------------------------------------

        results = model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )


        # ---------------------------------------------------------------
        # Convert YOLO detections into DetectionResult objects
        # ---------------------------------------------------------------

        detections = []

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                coords = box.xyxy[0]

                x1 = float(coords[0])
                y1 = float(coords[1])
                x2 = float(coords[2])
                y2 = float(coords[3])

                detection = analyze_detection(
                    track_id=0,
                    object_name=object_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_width=float(frame_width),
                    frame_height=float(frame_height),
                )

                if detection is not None:
                    detections.append(detection)


        # ---------------------------------------------------------------
        # Detection reliability filter
        # ---------------------------------------------------------------

        stable = detector_filter.process_frame(detections)


        # ---------------------------------------------------------------
        # Build a simple stable-state signature
        #
        # Only object names are used here.
        # Confidence changes do NOT cause repeated terminal messages.
        # ---------------------------------------------------------------

        current_stable = tuple(
            sorted(item.object_name for item in stable)
        )


        # ---------------------------------------------------------------
        # Print ONLY when stable objects change
        # ---------------------------------------------------------------

        if current_stable != last_stable:

            if stable:

                print(
                    "[STABLE]",
                    [
                        f"{item.object_name} "
                        f"(confidence={item.confidence:.2f})"
                        for item in stable
                    ],
                )

            else:

                print("[STABLE] No confirmed object.")

            last_stable = current_stable


        # ---------------------------------------------------------------
        # Display camera with YOLO annotations
        # ---------------------------------------------------------------

        annotated = results[0].plot()

        cv2.imshow(
            "Detection Filter Test",
            annotated,
        )


        # ---------------------------------------------------------------
        # Quit
        # ---------------------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Q pressed.")
            break


finally:

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] Camera released.")
    print("[INFO] Detection filter test stopped.")
