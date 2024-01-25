import cv2

cap = cv2.VideoCapture('http://192.168.0.103:5000/camera_feed/front')

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame is empty or not correctly received.")
        continue  # Skip this iteration

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break