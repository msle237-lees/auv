import os
import sys
import cv2
import serial
import traceback
import threading
import subprocess
import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, Response


# Port Configuration (Linux)
def is_port_open(port):
    """Check if a port is open."""
    result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
    return str(port) in result.stdout

def open_port(port):
    """Open a port using iptables."""
    subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', str(port), '-j', 'ACCEPT'])

port = 5000

if not is_port_open(port):
    print(f"Port {port} is not open. Opening port...")
    open_port(port)
else:
    print(f"Port {port} is already open.")

# Flask Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
app.config['SQLALCHEMY_BINDS'] = {
    'OUTPUT_CONTROLS': 'sqlite:///output_controls.db',
    'INPUT_CONTROLS': 'sqlite:///input_controls.db',
    'SENSORS_INPUT': 'sqlite:///sensors_input.db'
}

db = SQLAlchemy(app)
    
class OutputControlDB(db.Model):
    """Database model class for storing output controls."""
    __bind_key__ = 'OUTPUT_CONTROLS'
    id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.DateTime, default=datetime.utcnow)
    M1 = db.Column(db.Integer)
    M2 = db.Column(db.Integer)
    M3 = db.Column(db.Integer)
    M4 = db.Column(db.Integer)
    M5 = db.Column(db.Integer)
    M6 = db.Column(db.Integer)
    M7 = db.Column(db.Integer)
    M8 = db.Column(db.Integer)
    Claw = db.Column(db.Integer)

    def __repr__(self):
        return f'<Output Control Data: {self.id}>'
    
class InputControlDB(db.Model):
    __bind_key__ = 'INPUT_CONTROLS'
    id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.DateTime, default=datetime.utcnow)
    X = db.Column(db.Float)
    Y = db.Column(db.Float)
    Z = db.Column(db.Float)
    Pitch = db.Column(db.Float)
    Roll = db.Column(db.Float)
    Yaw = db.Column(db.Float)
    Claw = db.Column(db.Integer)

    def __repr__(self):
        return f'<Input Control Data: {self.id}>'
    
class SensorsInputDB(db.Model):
    __bind_key__ = 'SENSORS_INPUT'
    id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.DateTime, default=datetime.utcnow)
    OTemp = db.Column(db.Float)
    TTube = db.Column(db.Float)
    Depth = db.Column(db.Float)
    Humidity = db.Column(db.Float)
    Voltage = db.Column(db.Float)
    Current = db.Column(db.Float)

    def __repr__(self):
        return f'<Sensors Input Data: {self.id}>'
    
class CameraPackage:
    """Class to handle camera-related functions."""

    def __init__(self, camera_index):
        """
        Initialize the CameraPackage.

        Args:
            camera_index (int): Index of the camera.
        """
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()
        self.running = False

    def start_camera(self):
        """Start the camera stream."""
        with self.lock:
            if not self.running:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.running = True

    def stop_camera(self):
        """Stop the camera stream."""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.running = False

    def get_frame(self):
        """
        Capture a frame from the camera.

        Returns:
            tuple: A tuple containing a boolean indicating success and the captured frame.
        """
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
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
        print(f"Saved frame as {filename}")

    def delete_image(self, filename):
        """
        Delete an image file.

        Args:
            filename (str): The name of the file to be deleted.
        """
        os.remove(os.path.join(self.frame_directory, filename))
        print(f"Deleted image {filename}")

    def release(self):
        """Release the camera resource."""
        self.cap.release()

class HardwarePackage:
    """Class to handle hardware interface and serial communication."""

    def __init__(self, port, baudrate=9600):
        """
        Initialize the HardwarePackage.

        Args:
            port (str): Serial port to use.
            baudrate (int): Baud rate for serial communication.
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.thread = threading.Thread(target=self.read_from_serial)
        self.thread.daemon = True

    def read_from_serial(self):
        """Continuously read data from the serial port and process it."""
        while True:
            data = self.get_data()
            if data:
                parsed_data = self.parse_data(data)
                self.save_data(parsed_data)

    def get_data(self):
        """
        Read data from the serial port.

        Returns:
            str: The read data, or None if there's an error or if the port is closed.
        """
        if self.ser.isOpen():
            try:
                return self.ser.readline().decode('utf-8').strip()
            except serial.SerialException:
                return None
        return None

    def parse_data(self, data):
        """
        Parse the raw data from the serial port.

        Args:
            data (str): Raw data string from the serial port.

        Returns:
            list: Parsed data as a list of values.
        """
        return data.split(',')

    def save_data(self, parsed_data):
        """
        Save parsed data into the database.

        Args:
            parsed_data (list): Parsed data to be saved.
        """
        new_entry = SensorsInputDB(
            Date=datetime.utcnow(),
            OTemp=float(parsed_data[15]),
            TTube=float(parsed_data[16]),
            Depth=float(parsed_data[17]),
            Humidity=float(parsed_data[18]),
            Voltage=float(parsed_data[19]),
            Current=float(parsed_data[20])
        )
        with app.app_context():
            db.session.add(new_entry)
            db.session.commit()

    def send_data(self, data):
        """
        Send data to the serial port.

        Args:
            data (str): Data to be sent to the serial device.
        """
        if self.ser.isOpen():
            try:
                self.ser.write(data.encode())
            except serial.SerialException:
                pass

    def start(self):
        """Start the hardware interface."""
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        """Stop the hardware interface."""
        if self.ser.isOpen():
            self.ser.close()

class NeuralNetworkPackage:
    def __init__(self):
        pass

    def get_data(self):
        pass

    def parse_data(self, data):
        pass

    def save_data(self):
        pass

    def delete_data(self):
        pass

class MovementPackage:
    def __init__(self):
        pass

    def get_data(self):
        pass

    def parse_data(self, data):
        pass

    def save_data(self):
        pass

    def delete_data(self):
        pass

class ControllerPackage:
    def __init__(self):
        pass

    def get_data(self):
        pass

    def parse_data(self, data):
        pass

    def map_data(self, data) -> dict:
        pass

    def save_data(self):
        pass

    def delete_data(self):
        pass

def capture_camera(camera):
    """
    Generator to capture frames from the camera.

    Args:
        camera (CameraPackage): Camera package object.

    Yields:
        bytes: JPEG encoded frame bytes.
    """
    while camera.running:
        success, frame = camera.get_frame()
        if not success:
            continue
        frame_bytes = camera.parse_frame(frame)
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    """Route to serve the main page."""
    return render_template('index.html')

# Camera instances
camera0 = CameraPackage(0)
camera1 = CameraPackage(1)

# Hardware instance
hardware = HardwarePackage('/dev/ttyACM0', 115200)

@app.route('/start_camera/<int:camera_index>')
def start_camera(camera_index):
    """
    Route to start a specific camera.

    Args:
        camera_index (int): Index of the camera to start.

    Returns:
        str: Status message.
    """
    camera = camera0 if camera_index == 0 else camera1
    camera.start_camera()
    return "Camera started", 200

@app.route('/stop_camera/<int:camera_index>')
def stop_camera(camera_index):
    """
    Route to stop a specific camera.

    Args:
        camera_index (int): Index of the camera to stop.

    Returns:
        str: Status message.
    """
    camera = camera0 if camera_index == 0 else camera1
    camera.stop_camera()
    return "Camera stopped", 200

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """
    Route to get the video feed of a specific camera.

    Args:
        camera_index (int): Index of the camera for the video feed.

    Returns:
        Response: Flask response object for video streaming.
    """
    camera = camera0 if camera_index == 0 else camera1
    return Response(capture_camera(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_hardware')
def start_hardware():
    """Route to start the hardware interface."""
    try:
        hardware.start()
        return "Hardware started", 200
    except Exception as e:
        return str(e), 500

@app.route('/stop_hardware')
def stop_hardware():
    """Route to stop the hardware interface."""
    try:
        hardware.stop()
        return "Hardware stopped", 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # Push an application context manually
    with app.app_context():
        # Create tables in the database
        db.create_all()

    app.run(debug=True)