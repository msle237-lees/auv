import cv2

cap = cv2.VideoCapture('http://192.168.0.103/video_feed/0')

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break