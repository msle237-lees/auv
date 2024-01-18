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

@app.route('/upload_image', methods=['POST'])
def upload_image():
    camera_type = request.args.get('camera', 'front')  # default to 'front' if not specified
    if camera_type not in latest_images:
        return "Invalid camera type", 400

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
