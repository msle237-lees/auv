import cv2
import numpy as np
from ultralytics import YOLO

class Camera_Package:
    def __init__(self, model = 'yolov8n.pt'):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CV_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CV_PROP_FRAME_HEIGHT, 1080)

        self.image = None

        self.send_width = 480
        self.send_height = 270

        self.model = YOLO(model)

    def grab_image(self):
        ret, img = self.cam.read()
        if img is not None:
            return self.resize(img)
        else:
            return None

    def detect(self, img):
        results = self.model(img, stream = True)
        return results

    def resize(self, img):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if self.send_width is None and self.send_height is None:
            return image

        # We are resizing height if width is none
        if self.send_width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = self.send_height / float(h)
            dim = (int(w * r), self.send_height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the width and construct the dimensions
            r = self.send_width / float(w)
            dim = (self.send_width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)

    def run(self):
        pass

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cp = Camera_Package()
    cp.run()
