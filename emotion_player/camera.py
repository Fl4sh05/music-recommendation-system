import cv2


class CameraStream:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        # CAP_DSHOW avoids long startup on some Windows setups
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(index)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
