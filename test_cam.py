# import cv2
# import requests

# # URLs of the video feeds
# front_cam_feed_url = 'http://192.168.0.103:5000/front_cam_feed'
# down_cam_feed_url = 'http://192.168.0.103:5000/down_cam_feed'

# front_cam = cv2.VideoCapture(front_cam_feed_url)
# down_cam = cv2.VideoCapture(down_cam_feed_url)

# try:
#     while True:
#         # Read from the front camera feed
#         ret1, frame1 = front_cam.read()
#         if ret1:
#             cv2.imshow('Front Camera Feed', frame1)
        
#         # Read from the down camera feed
#         ret2, frame2 = down_cam.read()
#         if ret2:
#             cv2.imshow('Down Camera Feed', frame2)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     # Release the VideoCapture objects
#     front_cam.release()
#     down_cam.release()
#     cv2.destroyAllWindows()

import cv2
import requests
from PIL import Image
import numpy as np
import io

# URLs of the video feeds
front_cam_feed_url = 'http://192.168.0.103:5000/front_cam_feed'
down_cam_feed_url = 'http://192.168.0.103:5000/down_cam_feed'

def get_frame_from_stream(stream_url):
    stream = requests.get(stream_url, stream=True)
    bytes = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes += chunk
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame
    return None

try:
    while True:
        # Read from the front camera feed
        frame1 = get_frame_from_stream(front_cam_feed_url)
        if frame1 is not None:
            cv2.imshow('Front Camera Feed', frame1)
        
        # Read from the down camera feed
        frame2 = get_frame_from_stream(down_cam_feed_url)
        if frame2 is not None:
            cv2.imshow('Down Camera Feed', frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
