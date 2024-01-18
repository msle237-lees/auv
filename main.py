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

# PID class
def _clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    elif (upper is not None) and (value > upper):
        return upper
    elif (lower is not None) and (value < lower):
        return lower
    return value


class PID(object):
    """A simple PID controller."""

    def __init__(
        self,
        Kp=1.0,
        Ki=0.0,
        Kd=0.0,
        setpoint=0,
        sample_time=0.01,
        output_limits=(None, None),
        auto_mode=True,
        proportional_on_measurement=False,
        differential_on_measurement=True,
        error_map=None,
        time_fn=None,
        starting_output=0.0,
    ):
        """
        Initialize a new PID controller.

        :param Kp: The value for the proportional gain Kp
        :param Ki: The value for the integral gain Ki
        :param Kd: The value for the derivative gain Kd
        :param setpoint: The initial setpoint that the PID will try to achieve
        :param sample_time: The time in seconds which the controller should wait before generating
            a new output value. The PID works best when it is constantly called (eg. during a
            loop), but with a sample time set so that the time difference between each update is
            (close to) constant. If set to None, the PID will compute a new output value every time
            it is called.
        :param output_limits: The initial output limits to use, given as an iterable with 2
            elements, for example: (lower, upper). The output will never go below the lower limit
            or above the upper limit. Either of the limits can also be set to None to have no limit
            in that direction. Setting output limits also avoids integral windup, since the
            integral term will never be allowed to grow outside of the limits.
        :param auto_mode: Whether the controller should be enabled (auto mode) or not (manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on
            the input directly rather than on the error (which is the traditional way). Using
            proportional-on-measurement avoids overshoot for some types of systems.
        :param differential_on_measurement: Whether the differential term should be calculated on
            the input directly rather than on the error (which is the traditional way).
        :param error_map: Function to transform the error value in another constrained value.
        :param time_fn: The function to use for getting the current time, or None to use the
            default. This should be a function taking no arguments and returning a number
            representing the current time. The default is to use time.monotonic() if available,
            otherwise time.time().
        :param starting_output: The starting point for the PID's output. If you start controlling
            a system that is already at the setpoint, you can set this to your best guess at what
            output the PID should give when first calling it to avoid the PID outputting zero and
            moving the system away from the setpoint.
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = None
        self._last_output = None
        self._last_error = None
        self._last_input = None

        if time_fn is not None:
            # Use the user supplied time function
            self.time_fn = time_fn
        else:
            import time

            try:
                # Get monotonic time to ensure that time deltas are always positive
                self.time_fn = time.monotonic
            except AttributeError:
                # time.monotonic() not available (using python < 3.3), fallback to time.time()
                self.time_fn = time.time

        self.output_limits = output_limits
        self.reset()

        # Set initial state of the controller
        self._integral = _clamp(starting_output, output_limits)

    def __call__(self, input_, dt=None):
        """
        Update the PID controller.

        Call the PID controller with *input_* and calculate and return a control output if
        sample_time seconds has passed since the last update. If no new output is calculated,
        return the previous output instead (or None if no value has been calculated yet).

        :param dt: If set, uses this value for timestep instead of real time. This can be used in
            simulations when simulation time is different from real time.
        """
        if not self.auto_mode:
            return self._last_output

        now = self.time_fn()
        if dt is None:
            dt = now - self._last_time if (now - self._last_time) else 1e-16
        elif dt <= 0:
            raise ValueError('dt has negative value {}, must be positive'.format(dt))

        if self.sample_time is not None and dt < self.sample_time and self._last_output is not None:
            # Only update every sample_time seconds
            return self._last_output

        # Compute error terms
        error = self.setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)
        d_error = error - (self._last_error if (self._last_error is not None) else error)

        # Check if must map the error
        if self.error_map is not None:
            error = self.error_map(error)

        # Compute the proportional term
        if not self.proportional_on_measurement:
            # Regular proportional-on-error, simply set the proportional term
            self._proportional = self.Kp * error
        else:
            # Add the proportional error on measurement to error_sum
            self._proportional -= self.Kp * d_input

        # Compute integral and derivative terms
        self._integral += self.Ki * error * dt
        self._integral = _clamp(self._integral, self.output_limits)  # Avoid integral windup

        if self.differential_on_measurement:
            self._derivative = -self.Kd * d_input / dt
        else:
            self._derivative = self.Kd * d_error / dt

        # Compute final output
        output = self._proportional + self._integral + self._derivative
        output = _clamp(output, self.output_limits)

        # Keep track of state
        self._last_output = output
        self._last_input = input_
        self._last_error = error
        self._last_time = now

        return output

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'Kp={self.Kp!r}, Ki={self.Ki!r}, Kd={self.Kd!r}, '
            'setpoint={self.setpoint!r}, sample_time={self.sample_time!r}, '
            'output_limits={self.output_limits!r}, auto_mode={self.auto_mode!r}, '
            'proportional_on_measurement={self.proportional_on_measurement!r}, '
            'differential_on_measurement={self.differential_on_measurement!r}, '
            'error_map={self.error_map!r}'
            ')'
        ).format(self=self)

    @property
    def components(self):
        """
        The P-, I- and D-terms from the last computation as separate components as a tuple. Useful
        for visualizing what the controller is doing or when tuning hard-to-tune systems.
        """
        return self._proportional, self._integral, self._derivative

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (Kp, Ki, Kd)."""
        return self.Kp, self.Ki, self.Kd

    @tunings.setter
    def tunings(self, tunings):
        """Set the PID tunings."""
        self.Kp, self.Ki, self.Kd = tunings

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not."""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PID controller."""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        Enable or disable the PID controller, optionally setting the last output value.

        This is useful if some system has been manually controlled and if the PID should take over.
        In that case, disable the PID by setting auto mode to False and later when the PID should
        be turned back on, pass the last output variable (the control variable) and it will be set
        as the starting I-term when the PID is set to auto mode.

        :param enabled: Whether auto mode should be enabled, True or False
        :param last_output: The last output, or the control variable, that the PID should start
            from when going from manual mode to auto mode. Has no effect if the PID is already in
            auto mode.
        """
        if enabled and not self._auto_mode:
            # Switching from manual mode to auto, reset
            self.reset()

            self._integral = last_output if (last_output is not None) else 0
            self._integral = _clamp(self._integral, self.output_limits)

        self._auto_mode = enabled

    @property
    def output_limits(self):
        """
        The current output limits as a 2-tuple: (lower, upper).

        See also the *output_limits* parameter in :meth:`PID.__init__`.
        """
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Set the output limits."""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        if (None not in limits) and (max_output < min_output):
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        self._integral = _clamp(self._integral, self.output_limits)
        self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        """
        Reset the PID controller internals.

        This sets each term to 0 as well as clearing the integral, the last output and the last
        input (derivative calculation).
        """
        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._integral = _clamp(self._integral, self.output_limits)

        self._last_time = self.time_fn()
        self._last_output = None
        self._last_input = None

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
        return f'<Input Control Data: {self.Date}>'
    
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
    X = db.Column(db.Float)
    Y = db.Column(db.Float)
    Z = db.Column(db.Float)
    Pitch = db.Column(db.Float)
    Roll = db.Column(db.Float)
    Yaw = db.Column(db.Float)

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
        """
        Movement Package:
        - Handles the generation of motor and claw commands based on either controller input or neural network output.
        - All input comes from two databases, and all output is saved to a different database. 
        - The databases are updated every microsecond.
        - 
        """
        self.PIDs = []
        self.PWM_To_Force_Conversion = {}
        self.controller_data = {}
        self.neural_network_data = {}
        self.sensors_data = {}
        self.output_control_data_part_1 = [1500, 1500, 1500, 1500]
        self.output_control_data_part_2 = [1500, 1500, 1500, 1500]

        d1 = 0.3
        d2 = 0.15
        d3 = 0.2
        dh = np.sqrt(d1**2 + d3**2)
        dv = np.sqrt(d2**2 + d3**2)
        Fx = np.sin(45) * dh
        Fy = np.cos(45) * dv
        Fz = -1
        self.PID_Matrix_1 = np.array([
            [Fx, Fx, Fx, Fx],
            [Fy, Fy, Fy, Fy],
            [dh, dh, dh, dh]
        ]).transpose()
        self.PID_Matrix_2 = np.array([
            [Fz, Fz, Fz, Fz],
            [Fy, -Fy, -Fy, Fy],
            [Fx, -Fx, -Fx, Fx]
        ]).transpose()

        self.initialize_PIDs()
        self.initialize_PWM_To_Force_Conversion()

        self.in_min = -1
        self.in_max = 1
        self.out_min = 1300
        self.out_max = 1700

        movement_logger.info('Movement Package initialized')

    def initialize_PIDs(self):
        self.PIDs.append(PID(Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0, sample_time=0.01, output_limits=(None, None), auto_mode=True, proportional_on_measurement=False, differential_on_measurement=True, error_map=None, time_fn=None, starting_output=0.0))
        self.PIDs.append(PID(Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0, sample_time=0.01, output_limits=(None, None), auto_mode=True, proportional_on_measurement=False, differential_on_measurement=True, error_map=None, time_fn=None, starting_output=0.0))

    def initialize_PWM_To_Force_Conversion(self):
        df = pd.read_excel('static/data/T200-PWM-Force-Current.xlsx', engine='openpyxl')
        return df.to_dict()

    def get_sensors_data(self):
        data = SensorsInputDB.query.order_by(InputControlDB.Date.desc()).first()
        return {
            'Date': data.Date, 'X': data.X, 'Y': data.Y, 'Z': data.Z, 
            'Pitch': data.Pitch, 'Roll': data.Roll, 'Yaw': data.Yaw, 
            'Claw': data.Claw
        }

    def get_data(self):
        data = InputControlDB.query.order_by(InputControlDB.Date.desc()).first()
        return [data.X, data.Y, data.Z, data.Pitch, data.Roll, data.Yaw]

    def map_data(self, data):
        sensor_value = self.get_sensors_data() # Comes from sensors
        desired_value = [data[0], data[1], data[2], data[3], data[4], data[5]] # Comes from controller

        PID1_desired_data = [desired_value[0], desired_value[1], desired_value[5]]
        PID2_desired_data = [desired_value[2], desired_value[3], desired_value[4]]

        max_desired_PID1_index = max.index(PID1_desired_data)
        max_desired_PID2_index = max.index(PID2_desired_data)

        self.PIDs[0].setpoint = max(PID1_desired_data)
        self.PIDs[1].setpoint = max(PID2_desired_data)

        PID1_sensor_data = [sensor_value[0], sensor_value[1], sensor_value[5]]
        PID2_sensor_data = [sensor_value[2], sensor_value[3], sensor_value[4]]

        PID1_output = self.PIDs[0](PID1_sensor_data)
        PID2_output = self.PIDs[1](PID2_sensor_data)

        for i in range(4):
            self.output_control_data_part_1[i] = self.mapping(self.PID_Matrix_1[max_desired_PID1_index][i] * PID1_output)
            self.output_control_data_part_2[i] = self.mapping(self.PID_Matrix_2[max_desired_PID2_index][i] * PID2_output)
        
        self.save_data(self.output_control_data_part_1, self.output_control_data_part_2, data[6])

    def save_data(self, data1, data2, data3):
        output_data = OutputControlDB(
            Date=datetime.utcnow(),
            M1=self.mapping(data1[i]),
            M2=self.mapping(data1[i]),
            M3=self.mapping(data1[i]),
            M4=self.mapping(data1[i]),
            M5=self.mapping(data2[i]),
            M6=self.mapping(data2[i]),
            M7=self.mapping(data2[i]),
            M8=self.mapping(data2[i]),
            Claw=data3
        )
        db.session.add(output_data)
        db.session.commit()

    def delete_data(self):
        OutputControlDB.query.delete()
        db.session.commit()

    def mapping(self, x):
        return (x - self.in_min) * (self.out_max - self.out_min) / (self.in_max - self.in_min) + self.out_min

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
            Date=datetime.utcnow(),
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
camera0 = CameraPackage('http://192.168.0.103:5000/front_cam_feed')
camera1 = CameraPackage('http://192.168.0.103:5000/down_cam_feed')
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
            'B3Current': item.B3Current,
            'X': item.X,
            'Y': item.Y,
            'Z': item.Z,
            'Pitch': item.Pitch,
            'Roll': item.Roll,
            'Yaw': item.Yaw
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

# Create an instance of MovementPackage
movement_package = MovementPackage()

@app.route('/get-sensors-data', methods=['GET'])
def get_sensors_data():
    """Endpoint to get sensors data."""
    data = movement_package.get_sensors_data()
    return jsonify(data)

@app.route('/process-data', methods=['POST'])
def process_data():
    """Endpoint to process data."""
    # Assuming data is sent as JSON in the POST request
    input_data = request.json
    movement_package.map_data(input_data)
    return jsonify({"status": "Data processed successfully"})

@app.route('/delete-data', methods=['DELETE'])
def delete_data():
    """Endpoint to delete data."""
    movement_package.delete_data()
    return jsonify({"status": "Data deleted successfully"})

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
