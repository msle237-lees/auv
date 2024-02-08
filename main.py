from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

def test_camera_index(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera at index {index} not found")
        return False
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imshow(f"Camera Index {index}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Camera at index {index} is working")
        return True
    else:
        print(f"Camera at index {index} found but cannot retrieve frame")
        return False

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
    return render_template('index.html')

if __name__ == '__main__':
    # for i in range(10):  # Test indices 0-9
    #     test_camera_index(i)
    app.run(debug=True, threaded=True, host='0.0.0.0')
