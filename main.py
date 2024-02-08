from flask import Flask, Response, render_template
import cv2
import threading

app = Flask(__name__)

# Global variables to hold the camera threads
camera_threads = {}

def generate_frames(camera_index):
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FPS, 30)
    while camera_threads[camera_index]['run']:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

def start_camera_thread(camera_index):
    # Check if the thread for this camera is already running
    if camera_index in camera_threads and camera_threads[camera_index]['run']:
        return
    # Set up and start a new thread for the camera
    camera_threads[camera_index] = {'run': True, 'thread': threading.Thread(target=lambda: app.response_class(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame'))}
    camera_threads[camera_index]['thread'].start()

def stop_camera_thread(camera_index):
    if camera_index in camera_threads:
        # Signal the thread to stop
        camera_threads[camera_index]['run'] = False
        # Wait for the thread to finish
        camera_threads[camera_index]['thread'].join()

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    start_camera_thread(camera_index)
    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed/<int:camera_index>')
def stop_feed(camera_index):
    stop_camera_thread(camera_index)
    return "Camera feed stopped."

@app.route('/')
def index():
    # Ensure camera feeds are started for initial page load
    start_camera_thread(0)
    start_camera_thread(4)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
