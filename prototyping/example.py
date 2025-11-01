import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Speeds up OpenCV2 attaching to the a webcamera.
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import cv2

def convert_cv_to_qpixmap(cv_img):
    """ Converts opencv image into qpixmap which is optimized for display """
    height, width, _ = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img)

class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 640, 480)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.video_label)
        self.setCentralWidget(central_widget)

        self.cap = cv2.VideoCapture(0) # Initialize camera
        self.timer = self.startTimer(30) # Update every 30ms

    def timerEvent(self, event):
        """ Every call from timer updates the frame on screen. This is where we would process any data too."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            pixmap = convert_cv_to_qpixmap(frame)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release() # Release camera resources

if __name__ == '__main__':
    app = QApplication([])
    window = Example()
    window.show()
    app.exec()