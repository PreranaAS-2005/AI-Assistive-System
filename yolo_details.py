from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Run detection
    results = model(frame,conf=0.6)

    # Get all detected objects
    boxes = results[0].boxes

    # Print details of every detected object
    for box in boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        print(f"{object_name} : {confidence:.2f}")

    # Display detections
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Details", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()