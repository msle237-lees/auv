from flask import Flask, Response, render_template
import cv2
import threading
from time import sleep

app = Flask(__name__)

# Dictionary to store camera capture threads
camera_threads = {}
# Dictionary to store the latest frame from each camera
latest_frames = {}

def capture_frames(camera_index):
    global latest_frames
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FPS, 30)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    if not camera.isOpened():
        print(f"Failed to open camera with index {camera_index}.")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            print(f"Failed to capture frame from camera {camera_index}.")
            sleep(1)  # Sleep briefly and try again
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame)
        print(f'Camera {camera_index}: {len(buffer.tobytes())} bytes')
        latest_frames[camera_index] = buffer.tobytes()

    camera.release()

def start_camera_thread(camera_index):
    """Starts a separate thread for each camera to continuously capture frames."""
    if camera_index not in camera_threads:
        thread = threading.Thread(target=capture_frames, args=(camera_index,))
        camera_threads[camera_index] = thread
        thread.start()
        print(f"Started camera thread for camera {camera_index}.")

def generate_frames(camera_index):
    """Yield the latest frame captured by the specified camera."""
    global latest_frames
    while True:
        if camera_index in latest_frames:
            frame = latest_frames[camera_index]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            print(f'No frame available for camera {camera_index}.')
            # If no frame is available, yield an empty frame or placeholder
            sleep(0.1)  # Avoid tight loop if no frames are available

cam1 = '/dev/v4l/by-id/usb-Anker_PowerConf_C200_Anker_PowerConf_C200_ACNV9P0D07619591-video-index0'
cam2 = '/dev/v4l/by-id/usb-USB_Camera_USB_Camera-video-index0'

@app.route('/cam0_video_feed')
def video_feed():
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam1_video_feed')
def video_feed():
    return Response(generate_frames(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Optionally start camera threads for known cameras here
    return render_template('index.html')

if __name__ == '__main__':
    # Optionally pre-start camera threads for known camera indices
    start_camera_thread(0)
    start_camera_thread(1)
    app.run(debug=True, threaded=True, host='0.0.0.0')
