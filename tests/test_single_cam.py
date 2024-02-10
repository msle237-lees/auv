from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Failed to open camera with index {camera_index}.")
        return b''  # Return empty bytes if the camera fails to open

    while True:
        success, frame = camera.read()
        if not success:
            break  # Exit the loop if the camera fails to capture a frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """Video streaming route for a specific camera."""
    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Render a template if necessary or redirect to a specific camera feed
    return Response('Camera streaming server is running. Access the feed at /video_feed/{camera_index}')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
