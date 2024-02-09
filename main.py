from flask import Flask, Response, render_template, request
import cv2
import threading
from queue import Queue

app = Flask(__name__)

# Frame queues for each camera
frame_queues = {}

# Recording states and file handles for each camera
recording_states = {}
video_writers = {}

def camera_thread(camera_index):
    """
    Thread for capturing frames from a camera and managing recording state.
    
    @param camera_index: The index of the camera.
    """
    camera = cv2.VideoCapture(camera_index)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Use 'mp4v' as the codec and change file extension to '.mp4'
    video_format = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Check if we need to start recording
        if recording_states.get(camera_index, False):
            if camera_index not in video_writers:
                video_writers[camera_index] = cv2.VideoWriter(f'camera_{camera_index}.mp4', video_format, fps, (width, height))
            video_writers[camera_index].write(frame)

        if camera_index in frame_queues:
            if frame_queues[camera_index].full():
                frame_queues[camera_index].get()  # Remove oldest frame if full
            frame_queues[camera_index].put(frame)
    
    camera.release()
    # Close video writer if recording was enabled
    if camera_index in video_writers:
        video_writers[camera_index].release()
        del video_writers[camera_index]

@app.route('/control_recording/<int:camera_index>', methods=['POST'])
def control_recording(camera_index):
    """
    Starts or stops recording for the specified camera based on the POSTed action.
    
    @param camera_index: The index of the camera to control recording for.
    """
    action = request.form.get('action', 'stop')
    if action == 'start':
        recording_states[camera_index] = True
    elif action == 'stop':
        if camera_index in recording_states:
            recording_states[camera_index] = False
            # Close and remove the video writer for this camera
            if camera_index in video_writers:
                video_writers[camera_index].release()
                del video_writers[camera_index]
    return f"Recording {'started' if action == 'start' else 'stopped'} for camera {camera_index}"

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
