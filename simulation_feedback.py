from flask import Flask, request, Response
import threading
import io

app = Flask(__name__)

# Locks for thread-safe operations on the image data
image_locks = {
    'front': threading.Lock(),
    'down': threading.Lock()
}

# In-memory byte arrays to store the latest images
latest_images = {
    'front': io.BytesIO(),
    'down': io.BytesIO()
}

import socket
from threading import Thread

def start_tcp_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"TCP server listening on {host}:{port}...")
    
    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        Thread(target=handle_client, args=(conn,)).start()

def handle_client(conn):
    try:
        while True:
            # Receive camera ID first (1 byte)
            camera_id_data = conn.recv(1)
            if not camera_id_data:
                break
            
            camera_id = int.from_bytes(camera_id_data, byteorder='big')
            camera_type = 'front' if camera_id == 1 else 'down'

            # Receive image size (4 bytes)
            size_data = conn.recv(4)
            if not size_data:
                break

            image_size = int.from_bytes(size_data, byteorder='big')
            print(f"Expecting image of size: {image_size}")

            # Receive the actual image
            image_data = b''
            while len(image_data) < image_size:
                packet = conn.recv(image_size - len(image_data))
                if not packet:
                    break
                image_data += packet

            if image_data:
                with image_locks[camera_type]:
                    latest_images[camera_type].seek(0)
                    latest_images[camera_type].truncate(0)
                    latest_images[camera_type].write(image_data)
                    latest_images[camera_type].seek(0)
            else:
                print("No image data received.")
                break
    finally:
        conn.close()

# Start the TCP server in a separate thread
Thread(target=start_tcp_server, args=('0.0.0.0', 12345)).start()


@app.route('/upload_image', methods=['POST'])
def upload_image():
    camera_type = request.args.get('camera', 'front')  # default to 'front' if not specified

    file = request.files['image']
    if file:
        with image_locks[camera_type]:
            latest_images[camera_type].seek(0)  # Move to the start of the BytesIO object
            latest_images[camera_type].truncate(0)  # Clear previous image
            file.save(latest_images[camera_type])
            latest_images[camera_type].seek(0)  # Rewind the file pointer to the start
        return "Image received", 200
    else:
        return "No selected file", 400

@app.route('/camera_feed/<camera_type>')
def camera_feed(camera_type):
    if camera_type not in latest_images:
        return "Invalid camera type", 400

    def generate():
        while True:
            with image_locks[camera_type]:
                frame = latest_images[camera_type].getvalue()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'No Image' + b'\r\n\r\n')  # Serve a placeholder or an empty frame
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
