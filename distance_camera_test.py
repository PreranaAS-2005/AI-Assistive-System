from ultralytics import YOLO
import cv2

from navigation.spatial_analyzer import analyze_detection
from navigation.distance_estimator import DistanceEstimator


CAMERA_INDEX = 2
CONFIDENCE = 0.60
IMAGE_SIZE = 320


print("[INFO] Loading YOLO...")
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    print("[ERROR] Camera could not be opened.")
    raise SystemExit(1)

estimator = DistanceEstimator(
    focal_length_px=700.0
)

print("[INFO] Camera connected.")
print()
print("Show ONE object at a time.")
print("Test person, laptop, bottle and chair.")
print("Keep the object still.")
print("Press Q to quit.")
print()

last_report = None

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] Frame failed.")
            break

        height, width = frame.shape[:2]

        results = model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        current = []

        if results[0].boxes is not None:

            for box in results[0].boxes:

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                object_name = model.names[class_id]

                if confidence < CONFIDENCE:
                    continue

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
                    frame_width=float(width),
                    frame_height=float(height),
                )

                if detection is None:
                    continue

                estimate = estimator.estimate(
                    detection
                )

                if estimate is None:
                    continue

                current.append(
                    (
                        estimate.object_name,
                        round(estimate.distance_m, 2),
                        round(estimate.confidence, 2),
                    )
                )

        current.sort()

        state = tuple(current)

        if state != last_report:

            if current:

                print(
                    "[DISTANCE]",
                    [
                        f"{name}: {distance} m "
                        f"(confidence={confidence})"
                        for name, distance, confidence in current
                    ],
                )

            else:

                print(
                    "[DISTANCE] No supported object."
                )

            last_report = state

        annotated = results[0].plot()

        cv2.imshow(
            "Distance Test",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()

    cv2.destroyAllWindows()

    print("[INFO] Camera released.")
    print("[INFO] Distance test stopped.")