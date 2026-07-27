import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

while True:
    # Read one frame from the camera
    success, frame = camera.read()

    if not success:
        print("Failed to access the camera.")
        break

    # Display the frame
    cv2.imshow("AI Assistive System - Camera", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()