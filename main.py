from flask import Flask, Response, render_template
import pyzed.sl as sl
import cv2

app = Flask(__name__)

def generate_frames(camera_index):
    if camera_index == 1:  # Replace <ZED_CAMERA_INDEX> with the index you use for the ZED camera
        # ZED camera initialization and streaming logic
        zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER

        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera.")
            zed.close()
            return

        runtime_params = sl.RuntimeParameters()
        mat = sl.Mat()
        depth_map = sl.Mat()

        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(mat, sl.VIEW.LEFT)  # Get the left image
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Get the depth map

                # Convert the image to a format suitable for streaming
                frame = mat.get_data()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # For depth data, you would process and stream it similarly, potentially in a separate route
    else:
        # Existing logic for regular webcams
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FPS, 30)
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
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
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
