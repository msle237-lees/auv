# from flask import Flask, Response, render_template, request
# import cv2
# import threading
# from queue import Queue

# app = Flask(__name__)

# # Frame queues for each camera
# frame_queues = {}

# # Recording states and file handles for each camera
# recording_states = {}
# video_writers = {}

# def camera_thread(camera_index):
#     """
#     Thread for capturing frames from a camera and managing recording state.
    
#     @param camera_index: The index of the camera.
#     """
#     global frame_queues  # Ensure we're modifying the global dictionary

#     camera = cv2.VideoCapture(camera_index)
#     if not camera.isOpened():
#         print(f"Failed to open camera with index {camera_index}.")
#         return  # Exit the thread if camera cannot be opened

#     fps = camera.get(cv2.CAP_PROP_FPS)
#     width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     video_format = cv2.VideoWriter_fourcc(*'mp4v')

#     # Initialize the frame queue for this camera index if it doesn't exist
#     if camera_index not in frame_queues:
#         frame_queues[camera_index] = Queue(maxsize=10)  # Adjust maxsize as needed

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Check if we need to start recording
#         if recording_states.get(camera_index, False):
#             if camera_index not in video_writers:
#                 video_writers[camera_index] = cv2.VideoWriter(f'camera_{camera_index}.mp4', video_format, fps, (width, height))
#             video_writers[camera_index].write(frame)

#         if camera_index in frame_queues:
#             if frame_queues[camera_index].full():
#                 frame_queues[camera_index].get()  # Remove oldest frame if full
#             frame_queues[camera_index].put(frame)
    
#     camera.release()
#     # Close video writer if recording was enabled
#     if camera_index in video_writers:
#         video_writers[camera_index].release()
#         del video_writers[camera_index]

# def stream_video(camera_index):
#     """Generator function to stream video frames as JPEG."""
#     # Dynamically initialize the frame queue if it doesn't exist
#     if camera_index not in frame_queues:
#         frame_queues[camera_index] = Queue(maxsize=10)

#     while True:
#         if not frame_queues[camera_index].empty():
#             frame = frame_queues[camera_index].get()
#             _, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         else:
#             # Optionally, yield a placeholder frame or sleep briefly
#             pass

# @app.route('/video_feed/<int:camera_index>')
# def video_feed(camera_index):
#     """Route to stream video from a given camera."""
#     return Response(stream_video(camera_index),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/control_recording/<int:camera_index>', methods=['POST'])
# def control_recording(camera_index):
#     """
#     Starts or stops recording for the specified camera based on the POSTed action.
    
#     @param camera_index: The index of the camera to control recording for.
#     """
#     action = request.form.get('action', 'stop')
#     if action == 'start':
#         recording_states[camera_index] = True
#     elif action == 'stop':
#         if camera_index in recording_states:
#             recording_states[camera_index] = False
#             # Close and remove the video writer for this camera
#             if camera_index in video_writers:
#                 video_writers[camera_index].release()
#                 del video_writers[camera_index]
#     return f"Recording {'started' if action == 'start' else 'stopped'} for camera {camera_index}"

# @app.route('/')
# def index():
#     """Main page route."""
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True, threaded=True, host='0.0.0.0')

from flask import Flask, Response, render_template
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
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
