from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames(camera_index):
    camera = cv2.VideoCapture(camera_index)  # Dynamically use the camera index
    # Attempt to set the FPS to 30
    camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()  # Release the camera resource

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """Video streaming route for a specific camera."""
    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home page with links to video streams."""
    # Example with direct embedding
    return Response('''
    <html>
    <body>
    <h2>Camera 0</h2>
    <img src="/video_feed/0">
    <h2>Camera 4</h2>
    <img src="/video_feed/4">
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
