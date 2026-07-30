from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    # Run YOLO on the current frame
    results = model(frame)

    # Draw detection boxes on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Exit when 'x' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

camera.release()
cv2.destroyAllWindows()