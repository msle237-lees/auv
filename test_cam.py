import cv2
import requests

# URLs of the video feeds
front_cam_feed_url = 'http://192.168.0.103:5000/front_cam_feed'
down_cam_feed_url = 'http://192.168.0.103:5000/down_cam_feed'

# URLs for pressing the spacebar
start_spacebar_url = 'http://192.168.0.103:5000/start_pressing_spacebar'
stop_spacebar_url = 'http://192.168.0.103:5000/stop_pressing_spacebar'

# Function to start pressing the spacebar
def start_pressing_spacebar():
    try:
        response = requests.get(start_spacebar_url)
        print(response.text)
    except Exception as e:
        print(f"Error starting to press spacebar: {e}")

# Function to stop pressing the spacebar
def stop_pressing_spacebar():
    try:
        response = requests.get(stop_spacebar_url)
        print(response.text)
    except Exception as e:
        print(f"Error stopping spacebar press: {e}")

# Start pressing the spacebar
start_pressing_spacebar()

# Connect to the video feeds
front_cam = cv2.VideoCapture(front_cam_feed_url)
down_cam = cv2.VideoCapture(down_cam_feed_url)

try:
    while True:
        # Read from the front camera feed
        ret1, frame1 = front_cam.read()
        if ret1:
            cv2.imshow('Front Camera Feed', frame1)
        
        # Read from the down camera feed
        ret2, frame2 = down_cam.read()
        if ret2:
            cv2.imshow('Down Camera Feed', frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop pressing the spacebar
    stop_pressing_spacebar()

    # Release the VideoCapture objects
    front_cam.release()
    down_cam.release()
    cv2.destroyAllWindows()
