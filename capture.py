import cv2

# Try index 0 first, then 1 if that fails
cap = cv2.VideoCapture(0) 

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('capture.jpg', frame)
        print("Success: Saved to capture.jpg")
    else:
        print("Error: Camera found but could not read frame")
    cap.release()
else:
    print("Error: Could not open camera")