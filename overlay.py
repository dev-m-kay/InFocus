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

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.showFullScreen()

        self.drawn_point = QPointF(0,0)

        self.look_points = []
        self.look_point = QPoint(0,0)

        self.drawLook = True
        
        self.sample_index = 0
        self.repeat_index = 0
        self.sample_function = None
        self.samples = None

        self.sample_points = []

        self.isCalibrating = False
        
        self.not_looking_flag = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.nextPoint)  

        self.show()
    
    def updateLookPoint(self, x,y):

        self.look_points.append([x,y])

        if len(self.look_points) > 5:
            self.look_points.pop(0)

        count = 0
        ax = 0
        ay = 0
        for point in self.look_points:
            count += 1
            ax += point[0]
            ay += point[1]
    
        self.look_point = QPoint(int(ax/count),int(ay/count))
        self.update()

    def startCalibration(self,sample_points,sample_function,end_function):

        self.isCalibrating = True
        self.sample_points = sample_points
        self.sample_index = 0
        self.repeat_index = 0

        self.sample_function = sample_function
        self.end_function = end_function

        self.drawn_point = QPointF(*self.sample_points[self.sample_index])

        self.update()
        QTimer.singleShot(1000, self.nextSample)  # 1 second interval

    def nextSample(self):

        self.sample_function(self.sample_index)
        self.repeat_index += 1

        if self.repeat_index >= 9:

            QTimer.singleShot(100, self.nextPoint)
            return

        QTimer.singleShot(100, self.nextSample)
    
    def nextPoint(self):

        self.sample_function(self.sample_index)
        self.sample_index += 1
        print(self.sample_index)
        print(len(self.sample_points))

        if self.sample_index >= len(self.sample_points):
            self.end_function()
            self.isCalibrating = False
            self.timer.stop()
            self.update()
            return

        self.drawn_point = QPointF(*self.sample_points[self.sample_index])

        self.repeat_index = 0
        QTimer.singleShot(1000, self.nextSample)  # 1 second interval
        self.update()


    def paintEvent(self, event):
        painter = QPainter(self)
        if self.not_looking_flag:
            
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Example: Draw a semi-transparent red rectangle
            painter.setBrush(QColor(255, 0, 0, 100))  # RGBA color, 100 for alpha (transparency)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect().adjusted(50, 50, -50, -50)) # Adjust to draw within bounds

        if self.isCalibrating:
            painter.setBrush(QColor(255, 0, 0, 100))  # RGBA color, 100 for alpha (transparency)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.drawn_point, 20, 20)
        
        if self.drawLook:
            painter.setBrush(QColor(255, 255, 0, 100))  # RGBA color, 100 for alpha (transparency)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.look_point, 20, 20)

    def getCurrentGaze(self):
        return self.look_point.x(), self.look_point.y()

