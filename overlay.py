import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Speeds up OpenCV2 attaching to the a webcamera.
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
import mediapipe as mp
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
from PyQt6.QtCore import *
import cv2
import numpy as np

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowTransparentForInput  # Makes the overlay transparent to input
        )

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

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.showFullScreen()
        
        self.notLookingFlag = False

        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.notLookingFlag:
            
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Example: Draw a semi-transparent red rectangle
            painter.setBrush(QColor(255, 0, 0, 100))  # RGBA color, 100 for alpha (transparency)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect().adjusted(50, 50, -50, -50)) # Adjust to draw within bounds

        # painter.setBrush(QColor(255, 0, 0, 100))  # RGBA color, 100 for alpha (transparency)
        # painter.setPen(Qt.PenStyle.NoPen)
        # painter.drawEllipse(QPoint(int(self.sampleX), int(self.sampleY)), 20, 20)
