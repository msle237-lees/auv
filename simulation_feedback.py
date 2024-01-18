from flask import Flask, send_file, Response
import os
from time import sleep
import keyboard
import threading

app = Flask(__name__)

front_cam_folder = 'C:/Users/ksuau/AppData/LocalLow/Aquapack/Robotics/RoboSubSim/FrontCam'
down_cam_folder = 'C:/Users/ksuau/AppData/LocalLow/Aquapack/Robotics/RoboSubSim/DownCam'

def get_latest_image(folder_path):
    try:
        # List all files and sort them by name (assuming the names contain the index)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if files:
            return files[0]  # Return the latest file
        else:
            return None
    except Exception as e:
        print(f"Error getting the latest image: {e}")
        return None

@app.route('/front_cam_feed')
def front_cam_feed():
    def generate():
        while True:
            # Get the latest image
            latest_image_path = get_latest_image(front_cam_folder)
            if latest_image_path:
                with open(latest_image_path, 'rb') as f:
                    frame = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            sleep(0.1)  # Adjust the frame rate as needed

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/down_cam_feed')
def down_cam_feed():
    def generate():
        while True:
            # Get the latest image
            latest_image_path = get_latest_image(down_cam_folder)
            if latest_image_path:
                with open(latest_image_path, 'rb') as f:
                    frame = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            sleep(0.1)  # Adjust the frame rate as needed

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flag to control the spacebar pressing thread
pressing_spacebar = False

def press_spacebar_periodically():
    global pressing_spacebar
    while pressing_spacebar:
        keyboard.press_and_release('space')
        sleep(0.01)  # Sleep for 10ms

@app.route('/start_pressing_spacebar')
def start_pressing_spacebar():
    global pressing_spacebar
    if not pressing_spacebar:
        pressing_spacebar = True
        thread = threading.Thread(target=press_spacebar_periodically)
        thread.start()
        return "Started pressing spacebar every 10ms."
    else:
        return "Already pressing spacebar."

@app.route('/stop_pressing_spacebar')
def stop_pressing_spacebar():
    global pressing_spacebar
    pressing_spacebar = False
    return "Stopped pressing spacebar."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
