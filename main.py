import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Speeds up OpenCV2 attaching to the a webcamera.
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
import mediapipe as mp
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from settingsManager import settings as s

from tracker import gaze

def convert_cv_to_qpixmap(cv_img):
    """ Converts opencv image into qpixmap which is optimized for display """
    height, width, _ = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img)

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        # dark mode default palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        QApplication.setPalette(dark_palette)


        self.mp_face_mesh = mp.solutions.face_mesh
        self.faceMesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
        self.setGeometry(100, 100, 640, 480)
        central_container = QWidget()
        central_container.setFixedSize(640, 480)
        self.setCentralWidget(central_container)
        central_container.setContentsMargins(0,0,0,0)

        video_container = QWidget(central_container)
        video_container.setGeometry(0, 0, 640, 480)
        video_container.setStyleSheet("background-color: #1e1e1e;")
        

        self.video_label = QLabel(video_container)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                border-radius: 8px;
            }
        """)
        self.video_label.setFixedSize(640, 480)

        self.settings_button = QPushButton("Settings", video_container)
        self.settings_button.setGeometry(640 - 130, 10, 100, 30)
        self.settings_button.clicked.connect(self.toggleSettings)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(50,50,50);
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgb(70,70,70);
            }
        """)
        self.settings_button.raise_()

        self.settings_panel = QWidget(central_container)
        self.settings_panel.setGeometry(0, 0, 640, 480)
        self.settings_panel.setContentsMargins(0,0,0,0)
        self.settings_panel.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
                border-radius: 10px;
            }
        """)

        
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setSpacing(5)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # widget styles
        # --------

        lineedit_style = """
            QLineEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 4px;
            }
            QLineEdit:focus {
                border: 1px solid #88c0d0;
            }
        """
        combobox_style = """
            QComboBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """
        button_style = """
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border-radius: 5px;
                padding: 6px 12px;
                margin: 0px 10px 5px 10px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """

        # --------

        # adds timer setting
        self.timer_setting = QLineEdit()
        row = self.make_row("Timer:", self.timer_setting)
        settings_layout.addWidget(row)

        # lookaway settingsd
        self.lookaway_setting = QLineEdit()
        row = self.make_row("Lookaway:", self.lookaway_setting)
        settings_layout.addWidget(row)

        #ignored areas settings
        self.ignored_setting = QComboBox()
        self.ignored_setting.addItem("None") # option 0
        self.ignored_setting.addItem("Left side") # option 1
        self.ignored_setting.addItem("Right side") # option 2

        row = self.make_row("Ignored Areas: ", self.ignored_setting)
        settings_layout.addWidget(row)



        self.setting_reset = QPushButton("Reset to Default")
        self.setting_reset.clicked.connect(self.resetSettings)

        self.settings_save = QPushButton("Save Settings")
        self.settings_save.clicked.connect(self.settingsUpdate)


        # add styles
        self.timer_setting.setStyleSheet(lineedit_style)
        self.lookaway_setting.setStyleSheet(lineedit_style)
        self.ignored_setting.setStyleSheet(combobox_style)

        self.setting_reset.setStyleSheet(button_style)
        self.settings_save.setStyleSheet(button_style)

        # Add buttons to settings layout
        settings_layout.addStretch(1)
        settings_layout.addWidget(self.setting_reset)
        settings_layout.addWidget(self.settings_save)


        self.settings_panel.setVisible(False)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.settings_panel)
        layout.addWidget(video_container)
        self.setCentralWidget(central_widget)

        # initialize everything else as before
        settings = s.read()
        self.timer_setting.setText(settings["timer"])
        self.lookaway_setting.setText(settings["lookaway"])
        self.ignored_setting.setCurrentIndex(settings["ignoredAreas"])
        self.cap = cv2.VideoCapture(0)
        self.timer = self.startTimer(30)

        self.show()

    def make_row(self, label_text, widget):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setStyleSheet("QLabel { font-size: 14px; padding: 10px; }")
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        widget.setFixedHeight(25)
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        row_layout.addWidget(label)
        row_layout.addWidget(widget)
        row_layout.addStretch(1)

        return row


    def timerEvent(self, event):
        """ Every call from timer updates the frame on screen. This is where we would process any data too."""
        ret, frame = self.cap.read()
        if ret:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            results = self.faceMesh.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                gaze(frame, results.multi_face_landmarks[0])  # gaze estimatio

   
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
        """Toggles between video and settings panel without changing window size."""
        if self.settings_panel.isVisible():
            self.settings_panel.setVisible(False)
            self.video_label.parent().setVisible(True)
            self.settings_button.setText("Settings")
        else:
            self.settings_panel.setVisible(True)
            self.video_label.parent().setVisible(False)
            self.settings_button.setText("Back")


    def settingsUpdate(self):
        """Handles changes when settings are updated."""
        settings = {
            "timer": self.timer_setting.text(),       # minutes
            "lookaway": self.lookaway_setting.text(), # seconds
            "ignoredAreas": self.ignored_setting.currentIndex(),
        }
        s.write(settings)
        self.settings_panel.setVisible(False)
        self.video_label.parent().setVisible(True)
        self.settings_button.setText("Settings")


    def resetSettings(self):
        settings = {
            "timer": "30",        # minutes
            "lookaway": "30",     # seconds
            "ignoredAreas": 0,
        }

        s.write(settings)
        self.timer_setting.setText(str(settings["timer"]))
        self.lookaway_setting.setText(str(settings["lookaway"]))
        self.ignored_setting.setCurrentIndex(settings["ignoredAreas"])
        self.settings_panel.setVisible(False)
        self.video_label.parent().setVisible(True)
        self.settings_button.setText("Settings")


    def closeEvent(self, event):
        self.cap.release() # Release camera resources

class Popup(QMainWindow):
    """Creates popup window."""
    def __init__(self, message="Time to refocus. Please return to work session.", position=(100, 100)):
        super().__init__()
        self.setGeometry(position[0], position[1], 400, 200)
        self.setWindowTitle("Popup")

        #styles

        button_style = """
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border-radius: 5px;
                padding: 6px 12px;
                margin: 0px 10px 5px 10px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """

        label = QLabel(message, self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setGeometry(0, 40, 400, 80)
        close_button = QPushButton("Close", self)
        close_button.setGeometry(150, 140, 100, 30)
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet(button_style)

        self.show()


if __name__ == '__main__':
    app = QApplication([])
    window = Example()
    app.exec()