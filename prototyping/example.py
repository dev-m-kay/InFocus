import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Speeds up OpenCV2 attaching to the a webcamera.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QLineEdit, QHBoxLayout, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from settingsManager import settings as s

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

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.toggleSettings)


        self.settings_panel = QWidget()
        settings_layout = QVBoxLayout(self.settings_panel)
        self.settings_save = QPushButton("Save Settings")
        self.settings_save.clicked.connect(self.settingsUpdate)

        # adds timer setting
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        timer_setting_label = QLabel("Timer: ")
        self.timer_setting = QLineEdit()

        row_layout.addWidget(timer_setting_label)
        row_layout.addWidget(self.timer_setting)
        settings_layout.addWidget(row)


        # lookaway settingsd
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        lookaway_setting_label = QLabel("lookaway: ")
        self.lookaway_setting = QLineEdit()

        row_layout.addWidget(lookaway_setting_label)
        row_layout.addWidget(self.lookaway_setting)
        settings_layout.addWidget(row)

        #ignored areas settings
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        ignored_setting_label = QLabel("Ignored areas: ")
        self.ignored_setting = QComboBox()
        self.ignored_setting.addItem("None") # option 0
        self.ignored_setting.addItem("Left side") # option 1
        self.ignored_setting.addItem("Right side") # option 2

        row_layout.addWidget(ignored_setting_label)
        row_layout.addWidget(self.ignored_setting)
        settings_layout.addWidget(row)




        self.setting_reset = QPushButton("Reset to Default")
        self.setting_reset.clicked.connect(self.resetSettings)
        self.settings_save = QPushButton("Save Settings")
        self.settings_save.clicked.connect(self.settingsUpdate)


        self.settings_panel.setVisible(False)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.settings_panel)
        layout.addWidget(self.video_label)



        settings_layout.addWidget(self.setting_reset)
        settings_layout.addWidget(self.settings_save)

        self.setCentralWidget(central_widget)


        settings = s.read()
        self.timer_setting.setText(settings["timer"])
        self.lookaway_setting.setText(settings["lookaway"])
        self.ignored_setting.setCurrentIndex(settings["ignoredAreas"])
        self.cap = cv2.VideoCapture(0) # Initialize camera
        self.timer = self.startTimer(30) # Update every 30ms

        

    def timerEvent(self, event):
        """ Every call from timer updates the frame on screen. This is where we would process any data too."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            pixmap = convert_cv_to_qpixmap(frame)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8) # creates blank frame
            frame = cv2.putText(
                frame, 
                'No webcam detected', 
                org=(50,50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(255, 255, 255), 
                thickness=2, 
                lineType=cv2.LINE_AA 
            )
            pixmap = convert_cv_to_qpixmap(frame)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def toggleSettings(self):
        """ toggles settings panel"""
        if self.settings_panel.isVisible():
            self.settings_panel.setVisible(False)
            self.video_label.setVisible(True)
            self.settings_button.setText("Settings")

        else:
            self.settings_panel.setVisible(True)
            self.video_label.setVisible(False)

    def settingsUpdate(self):
        """ handles changes when settings is updated. """
        settings = {
            "timer" : self.timer_setting.text(), #minutes
            "lookaway": self.lookaway_setting.text(), #seconds
            "ignoredAreas": self.ignored_setting.currentIndex(),
        }
        s.write(settings)
        self.cap = cv2.VideoCapture(0)
        self.settings_panel.setVisible(False)
        self.video_label.setVisible(True)

    def resetSettings(self):
        settings = {
            "timer": 30,        # minutes
            "lookaway": 30,     # seconds
            "ignoredAreas": 0,
        }

        s.write(settings)
        self.timer_setting.setText(str(settings["timer"]))
        self.lookaway_setting.setText(str(settings["lookaway"]))
        self.ignored_setting.setCurrentIndex(settings["ignoredAreas"])
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.settings_panel.setVisible(False)
        self.video_label.setVisible(True)



    def closeEvent(self, event):
        self.cap.release() # Release camera resources

class popup(QMainWindow):
     """ creates popup window"""
     def __init__(self, message, position):
        super().__init__()
        self.setGeometry(100, 100, 640, 480)
        self.setTabPosition()

if __name__ == '__main__':
    app = QApplication([])
    window = Example()
    window.show()
    app.exec()