import os
import sys
import cv2
import serial
import random
import logging
import traceback
import threading
import subprocess
import numpy as np
import pandas as pd
# from ultralytics import YOLOv8
from datetime import datetime
from flask import has_request_context
from flask_sqlalchemy import SQLAlchemy
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__, instance_relative_config=True)

# Ensure the instance folder exists
os.makedirs(app.instance_path, exist_ok=True)

class CameraPackage:
    """Class to handle camera-related functions."""

    def __init__(self, camera_index, camera_logger : logging.Logger):
        """
        Initialize the CameraPackage.

        Args:
            camera_index (int): Index of the camera.
        """
        self.camera_logger = camera_logger
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.frame_directory = 'static/imgs/frames'

        camera_logger.info('Camera Package initialized')

    def start_camera(self):
        """Start the camera stream."""
        with self.lock:
            if not self.running:
                try:
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_FFMPEG)
                    self.running = True

                    self.camera_logger.info(f'Camera {self.camera_index} started')
                except Exception as e:
                    self.camera_logger.error(f'Error starting camera {self.camera_index}: {e}')
                    self.running = False

    def stop_camera(self):
        """Stop the camera stream."""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.running = False

            self.camera_logger.info(f'Camera {self.camera_index} stopped')

    def start_recording(self, filename):
        """
        Start recording the video to a file.

        Args:
            filename (str): Filename for the recorded video.
        """
        with self.lock:
            if self.running and not hasattr(self, 'out'):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

                self.camera_logger.info(f'Camera {self.camera_index} recording started')

    def stop_recording(self):
        """
        Stop the video recording.
        """
        with self.lock:
            if hasattr(self, 'out'):
                self.out.release()
                del self.out
                self.camera_logger.info(f'Camera {self.camera_index} recording stopped')

    def get_frame(self):
        """
        Capture a frame from the camera and write to video file if recording.

        Returns:
            tuple: A tuple containing a boolean indicating success and the captured frame.
        """
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
            if ret and hasattr(self, 'out'):
                self.out.write(frame)
            return ret, frame

    def parse_frame(self, frame):
        """
        Encode the frame in JPEG format.

        Args:
            frame (numpy.ndarray): The frame to be encoded.

        Returns:
            bytes: The encoded frame bytes, or None if encoding fails.
        """
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes() if ret else None

    def save_image(self, frame):
        """
        Save the captured frame as an image file.

        Args:
            frame (numpy.ndarray): The frame to be saved.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.frame_directory, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self.camera_logger.info(f"Saved image {filename}")

    def delete_image(self, filename):
        """
        Delete an image file.

        Args:
            filename (str): The name of the file to be deleted.
        """
        os.remove(os.path.join(self.frame_directory, filename))
        print(f"Deleted image {filename}")
        self.camera_logger.info(f"Deleted image {filename}")

    def show_image(self, frame):
        """
        Display the frame in a window.

        Args:
            frame (numpy.ndarray): The frame to be displayed.
        """
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_camera()

    def release(self):
        """Release the camera resource and stop recording if it's on."""
        with self.lock:
            if self.cap:
                self.stop_recording()
                self.cap.release()
                self.camera_logger.info(f"Camera {self.camera_index} released")

# Create the other loggers for the other modules
def create_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.addHandler(RotatingFileHandler(filename, maxBytes=10000000, backupCount=5))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

camera_logger = create_logger('camera', 'static/logs/Camera.log')

camera1 = CameraPackage(0, camera_logger)
camera2 = CameraPackage(4, camera_logger)

# This shuts down the server and is used to ensure everything closes properly
def shutdown_server():
    print("Shutting down server...")
    if has_request_context():
        # Stop camera instances if the server is running in a request context
        camera1.stop_camera()
        camera2.stop_camera()
        # Stop the hardware interface
        # hardware.stop()
        # Stop the movement package
        # movement.stop()
        # Stop the neural network
        # neural_network.stop()

def capture_camera(camera):
    """
    Generator function to capture frames from a camera and encode them for HTTP streaming.

    This generator continuously captures frames from the specified camera, encodes them
    as JPEG for compatibility with web browsers, and yields them in a format suitable for
    HTTP multipart streaming. It uses the cv2.imencode function to convert frames into JPEG
    format, ensuring they can be efficiently transmitted over a network.

    @param camera An instance of a camera class that provides access to frame capture functionality.
    @yield Encoded frame data in multipart/x-mixed-replace format for live video streaming.
    """
    while camera.running:
        success, frame = camera.get_frame()
        if not success:
            continue
        # Encode the frame in JPEG format; may need to adjust parameters based on your camera's output
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = camera.parse_frame(buffer)
        camera.show_image(frame)
        if frame_bytes:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Create the necessary flask routes
# Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """
    Route to get the video feed of a specific camera.

    This function is designed to handle video streaming by selecting
    the appropriate camera based on the camera index. It dynamically
    generates a multipart response stream that updates with new frames
    from the selected camera.

    @param camera_index The index of the camera to stream.
    @return A Flask response object that streams the video feed.
    """
    camera = camera1 if camera_index == 0 else camera2
    return Response(capture_camera(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the flask app
if __name__ == '__main__':
    # Start camera instances
    camera1.start_camera()
    camera2.start_camera()

    camera1.show_image(camera1.parse_frame(camera1.get_frame()[1]))
    camera2.show_image(camera2.parse_frame(camera2.get_frame()[1]))

    try:
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)

    finally:
        # Ensure resources are released on shutdown
        shutdown_server()