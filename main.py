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
from yolov8 import YOLOv8
from datetime import datetime
from flask import has_request_context
from flask_sqlalchemy import SQLAlchemy
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, Response, request, jsonify

# Port Configuration (Linux)
def is_port_open(port):
    """Check if a port is open."""
    result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
    return str(port) in result.stdout

def open_port(port):
    """Open a port using iptables."""
    subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', str(port), '-j', 'ACCEPT'])

if not os.path.exists('static/logs'):
    os.makedirs('static/logs')

# Function to set up a logger for a specific module
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=1)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

# Function to set up a file logger
def setup_file_logger(name, log_file, level=logging.INFO):
    handler = RotatingFileHandler(log_file, maxBytes=10000, backupCount=1)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

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

def shutdown_server():
    print("Shutting down server...")
    if has_request_context():
        # Stop camera instances if the server is running in a request context
        camera0.stop_camera()
        camera1.stop_camera()

def generate_example_data():
    """Generate example data for testing UI."""
    for i in range(500):  # Insert 10 entries for each model
        # Creating OutputControlDB example data
        output_data = OutputControlDB(
            M1=1000+(2*i),
            M2=1000+(2*i),
            M3=1000+(2*i),
            M4=1000+(2*i),
            M5=1000+(2*i),
            M6=1000+(2*i),
            M7=1000+(2*i),
            M8=1000+(2*i),
            Claw=1000+(2*i)
        )
        db.session.add(output_data)


        if i >= 250: 
            Temp=1 
        else:
            Temp=0
        # Creating InputControlDB example data
        input_data = InputControlDB(
            X=-1+(0.004008*i),
            Y=-1+(0.004008*i),
            Z=-1+(0.004008*i),
            Pitch=-1+(0.004008*i),
            Roll=-1+(0.004008*i),
            Yaw=-1+(0.004008*i),
            Claw=Temp
        )
        db.session.add(input_data)

        # Creating SensorsInputDB example data
        sensor_data = SensorsInputDB(
            OTemp=-1+(0.1002*i),  # Example range for temperature
            TTube=-1+(0.1002*i),
            Depth=-1+(0.2004001*i),  # Example range for depth
            Humidity=-1+(0.2004001*i),
            Pressure=-1+(0.2004001*i),
            B1Voltage=-1+(0.036072*i),
            B2Voltage=-1+(0.036072*i),
            B3Voltage=-1+(0.036072*i),
            B1Current=-1+(0.06012*i),
            B2Current=-1+(0.06012*i),
            B3Current=-1+(0.06012*i)
        )
        db.session.add(sensor_data)

    db.session.commit()

# port = 5000

# if not is_port_open(port):
#     print(f"Port {port} is not open. Opening port...")
#     open_port(port)
# else:
#     print(f"Port {port} is already open.")

# Flask Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
app.config['SQLALCHEMY_BINDS'] = {
    'OUTPUT_CONTROLS': 'sqlite:///output_controls.db',
    'INPUT_CONTROLS': 'sqlite:///input_controls.db',
    'SENSORS_INPUT': 'sqlite:///sensors_input.db'
}

db = SQLAlchemy(app)

# Setup loggers for each package
camera_logger = setup_logger('camera', 'static/logs/camera_package.log')
hardware_logger = setup_logger('hardware', 'static/logs/hardware_package.log')
neural_network_logger = setup_logger('neural_network', 'static/logs/neural_network_package.log')
movement_logger = setup_logger('movement', 'static/logs/movement_package.log')
controller_logger = setup_logger('controller', 'static/logs/controller_package.log')

# Set up file logger for Flask's app logger
setup_file_logger('flask_server', 'static/logs/server.log')

# Ensure that the internal Flask log messages go to the file log
for handler in app.logger.handlers:
    app.logger.removeHandler(handler)
app.logger.addHandler(setup_file_logger('flask_server', 'static/logs/server.log'))


class OutputControlDB(db.Model):
    """Database model class for storing output controls."""
    __bind_key__ = 'OUTPUT_CONTROLS'
    Date = db.Column(db.DateTime, default=datetime.utcnow, primary_key=True)
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
    Date = db.Column(db.DateTime, default=datetime.utcnow, primary_key=True)
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
    Date = db.Column(db.DateTime, default=datetime.utcnow, primary_key=True)
    OTemp = db.Column(db.Float)
    TTube = db.Column(db.Float)
    Depth = db.Column(db.Float)
    Humidity = db.Column(db.Float)
    Pressure = db.Column(db.Float)
    B1Voltage = db.Column(db.Float)
    B2Voltage = db.Column(db.Float)
    B3Voltage = db.Column(db.Float)
    B1Current = db.Column(db.Float)
    B2Current = db.Column(db.Float)
    B3Current = db.Column(db.Float)

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

        camera_logger.info('Camera Package initialized')

    def start_camera(self):
        """Start the camera stream."""
        with self.lock:
            if not self.running:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.running = True

                camera_logger.info(f'Camera {self.camera_index} started')

    def stop_camera(self):
        """Stop the camera stream."""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.running = False

            camera_logger.info(f'Camera {self.camera_index} stopped')

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

                camera_logger.info(f'Camera {self.camera_index} recording started')

    def stop_recording(self):
        """
        Stop the video recording.
        """
        with self.lock:
            if hasattr(self, 'out'):
                self.out.release()
                del self.out
                camera_logger.info(f'Camera {self.camera_index} recording stopped')

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
        camera_logger.info(f"Saved image {filename}")

    def delete_image(self, filename):
        """
        Delete an image file.

        Args:
            filename (str): The name of the file to be deleted.
        """
        os.remove(os.path.join(self.frame_directory, filename))
        print(f"Deleted image {filename}")
        camera_logger.info(f"Deleted image {filename}")

    def release(self):
        """Release the camera resource and stop recording if it's on."""
        with self.lock:
            if self.cap:
                self.stop_recording()
                self.cap.release()
                camera_logger.info(f"Camera {self.camera_index} released")

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
        hardware_logger.info('Hardware Package initialized')

    def read_from_serial(self):
        """Continuously read data from the serial port and process it."""
        while True:
            data = self.get_data()
            if data:
                parsed_data = self.parse_data(data)
                self.save_data(parsed_data)
                camera_logger.info(f"Received data: {parsed_data}")

    def get_data(self):
        """
        Read data from the serial port.

        Returns:
            str: The read data, or None if there's an error or if the port is closed.
        """
        if self.ser.isOpen():
            try:
                hardware_logger.info('Reading data from serial port')
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
        hardware_logger.info(f"Saving data: {new_entry}")
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
                hardware_logger.info(f"Sending data: {data}")
                self.ser.write(data.encode())
            except serial.SerialException:
                pass

    def start(self):
        """Start the hardware interface."""
        if not self.thread.is_alive():
            hardware_logger.info('Starting hardware interface')
            self.thread.start()


    def stop(self):
        """Stop the hardware interface."""
        if self.ser.isOpen():
            hardware_logger.info('Stopping hardware interface')
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
        controller_logger.info('Controller Package initialized')

    def get_data(self):
        """
        Extracts data from the HTTP POST request.
        """
        controller_logger.info('Getting data from HTTP POST request')
        return request.json

    def parse_data(self, data):
        """
        Parses the data from the request. Adjust this as per your data structure.
        """
        # Example: Expecting data to be a dictionary with specific keys
        controller_logger.info('Parsing data from HTTP POST request')
        return {
            'X': data.get('X'),
            'Y': data.get('Y'),
            'Z': data.get('Z'),
            'Pitch': data.get('Pitch'),
            'Roll': data.get('Roll'),
            'Yaw': data.get('Yaw'),
            'Claw': data.get('Claw')
        }

    def save_data(self, data):
        """
        Saves the data to the database.
        """
        control_data = InputControlDB(
            X=data['X'], 
            Y=data['Y'], 
            Z=data['Z'], 
            Pitch=data['Pitch'], 
            Roll=data['Roll'], 
            Yaw=data['Yaw'], 
            Claw=data['Claw']
        )
        controller_logger.info(f'Saving data: {control_data}')
        db.session.add(control_data)
        db.session.commit()

    def delete_data(self):
        """
        Deletes all data from the database (be cautious with this).
        """
        controller_logger.info('Deleting all data from database')
        InputControlDB.query.delete()
        db.session.commit()

# Flask Routes
@app.route('/')
def index():
    """Route to serve the main page."""
    return render_template('index.html')

# Camera instances
camera0 = CameraPackage(0)
camera1 = CameraPackage(1)
camera_logger.info('Camera instances initialized')

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

@app.route('/start_recording/<int:camera_index>/')
def start_recording(camera_index):
    camera = camera0 if camera_index == 0 else camera1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "static/Videos/" + timestamp + ".avi"
    camera.start_recording(filename)
    return "Recording started", 200

@app.route('/stop_recording/<int:camera_index>')
def stop_recording(camera_index):
    camera = camera0 if camera_index == 0 else camera1
    camera.stop_recording()
    return "Recording stopped", 200

# Hardware instance
# hardware = HardwarePackage('/dev/ttyACM0', 115200)

# @app.route('/start_hardware')
# def start_hardware():
#     """Route to start the hardware interface."""
#     try:
#         hardware.start()
#         return "Hardware started", 200
#     except Exception as e:
#         return str(e), 500

# @app.route('/stop_hardware')
# def stop_hardware():
#     """Route to stop the hardware interface."""
#     try:
#         hardware.stop()
#         return "Hardware stopped", 200
#     except Exception as e:
#         return str(e), 500
    
# Controller instance
controller_package = ControllerPackage()

@app.route('/receive-data', methods=['POST'])
def receive_controller_data():
    data = controller_package.get_data()
    parsed_data = controller_package.parse_data(data)
    controller_package.save_data(parsed_data)
    return jsonify({"message": f"Controller data received and saved successfully"})

@app.route('/get-output-control-data')
def get_output_control_data():
    data = OutputControlDB.query.all()
    return jsonify([{
        'Date': item.Date, 'M1': item.M1, 'M2': item.M2, 'M3': item.M3, 
        'M4': item.M4, 'M5': item.M5, 'M6': item.M6, 'M7': item.M7, 
        'M8': item.M8, 'Claw': item.Claw
    } for item in data])

@app.route('/get-input-control-data')
def get_input_control_data():
    data = InputControlDB.query.all()
    return jsonify([{
        'Date': item.Date, 'X': item.X, 'Y': item.Y, 'Z': item.Z, 
        'Pitch': item.Pitch, 'Roll': item.Roll, 'Yaw': item.Yaw, 
        'Claw': item.Claw
    } for item in data])

@app.route('/get-sensors-input-data')
def get_sensors_input_data():
    data = SensorsInputDB.query.all()
    results = []
    for item in data:
        # Calculate the average voltage and current for each item
        BatteryV = (item.B1Voltage + item.B2Voltage + item.B3Voltage) / 3
        BatteryC = (item.B1Current + item.B2Current + item.B3Current) / 3
        
        results.append({
            'Date': item.Date.strftime("%Y-%m-%d %H:%M:%S"),  # Format the date as a string
            'OTemp': item.OTemp,
            'TTube': item.TTube,
            'Depth': item.Depth,
            'Humidity': item.Humidity,
            'Pressure': item.Pressure,
            'Voltage': BatteryV,
            'Current': BatteryC,
            'B1Voltage': item.B1Voltage,
            'B2Voltage': item.B2Voltage,
            'B3Voltage': item.B3Voltage,
            'B1Current': item.B1Current,
            'B2Current': item.B2Current,
            'B3Current': item.B3Current
        })
    
    return jsonify(results)

@app.route('/get-file-tree')
def get_file_tree():
    # Assuming 'logs' directory is in the same directory as this script
    files = os.listdir('static/logs')
    return jsonify({'files': files})

@app.route('/get-file-content')
def get_file_content():
    file_path = request.args.get('file')
    content = ''
    with open(f'static/logs/{file_path}', 'r') as file:
        content = file.read()
    return content

@app.route('/Start_Program')
def Start_Program():
    pass

@app.route('/Stop_Program')
def Stop_Program():
    pass

@app.route('/Reset_Program')
def Reset_Program():
    pass

if __name__ == '__main__':
    # Push an application context manually
    with app.app_context():
        # Create tables in the database
        db.create_all()
        # generate_example_data()

    # Start camera instances
    camera0.start_camera()
    camera1.start_camera()

    try:
        # Start Flask app
        app.run(debug=True)
    finally:
        # Ensure resources are released on shutdown
        shutdown_server()
        handlers = camera_logger.handlers + hardware_logger.handlers + \
                neural_network_logger.handlers + movement_logger.handlers + \
                controller_logger.handlers
        for handler in handlers:
            handler.close()
            camera_logger.removeHandler(handler)
