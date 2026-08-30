# import cv2

# # Open the default webcam
# camera = cv2.VideoCapture(0)

# while True:
#     # Read one frame from the camera
#     success, frame = camera.read()

#     if not success:
#         print("Failed to access the camera.")
#         break

#     # Display the frame
#     cv2.imshow("AI Assistive System - Camera", frame)

#     # Press 'x' to exit
#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break

# # Release the camera
# camera.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()




import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ Could not open laptop camera")
    exit()

print("✅ Camera opened successfully")

while True:
    success, frame = camera.read()

    if not success:
        break

    cv2.imshow("Laptop Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()