from flask import Flask, Response, render_template
import cv2
import threading
from queue import Queue

app = Flask(__name__)

# Each camera stream will have its own frame queue.
frame_queues = {}

def camera_thread(camera_index):
    """
    Camera thread function to capture frames and put them in a queue.
    
    @param camera_index The index of the camera to capture frames from.
    """
    if camera_index not in frame_queues:
        frame_queues[camera_index] = Queue(maxsize=10)  # Limit queue size to prevent memory issues

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FPS, 30)

    while True:
        success, frame = camera.read()
        if not success:
            break
        if frame_queues[camera_index].full():
            frame_queues[camera_index].get()  # Remove oldest frame if the queue is full
        frame_queues[camera_index].put(frame)
    
    camera.release()

def generate_frames(camera_index):
    """
    Generates camera frames from the queue for streaming.
    
    @param camera_index The index of the camera to generate frames from.
    """
    while True:
        if camera_index in frame_queues and not frame_queues[camera_index].empty():
            frame = frame_queues[camera_index].get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """
    Video streaming route. Starts the camera thread if not already started and streams the frames.
    
    @param camera_index The index of the camera for which the video feed is requested.
    """
    if camera_index not in frame_queues:
        # Start the camera thread
        t = threading.Thread(target=camera_thread, args=(camera_index,))
        t.daemon = True  # Daemon thread exits when the main thread exits
        t.start()

    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
