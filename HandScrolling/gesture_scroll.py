# === overlay_scroll_combined.py ===
import sys
import os
import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
from PyQt6 import QtCore, QtGui, QtWidgets

# Gesture detector class
class GestureScroll:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.drawer = mp.solutions.drawing_utils
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.15

    def is_only_index_extended(self, hand_landmarks):
        lm = hand_landmarks.landmark
        index_extended = lm[8].y < lm[6].y - 0.03
        middle_folded = lm[12].y > lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y
        thumb_folded = lm[4].x > lm[2].x
        return index_extended and middle_folded and ring_folded and pinky_folded and thumb_folded

    def is_only_index_down(self, hand_landmarks):
        lm = hand_landmarks.landmark
        index_down = lm[8].y > lm[6].y + 0.06 and lm[8].y > lm[7].y + 0.03
        middle_folded = lm[12].y > lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y
        thumb_folded = lm[4].x > lm[2].x
        return index_down and middle_folded and ring_folded and pinky_folded and thumb_folded

    def should_scroll(self):
        now = time.time()
        if now - self.last_scroll_time > self.scroll_cooldown:
            self.last_scroll_time = now
            return True
        return False

    def process(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for i, handLms in enumerate(result.multi_hand_landmarks):
                self.drawer.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
                hand_label = result.multi_handedness[i].classification[0].label

                if hand_label == "Right" and self.is_only_index_extended(handLms):
                    if self.should_scroll():
                        pyautogui.scroll(40)
                        print("Right hand up - scroll up")

                elif hand_label == "Left" and self.is_only_index_down(handLms):
                    if self.should_scroll():
                        pyautogui.scroll(-40)
                        print("Left hand down - scroll down")
        return frame

# PyQt Overlay Window
class CameraOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureCam")
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setStyleSheet("border-radius: 20px; background-color: black;")

        self.toggle_button = QtWidgets.QPushButton("👁 Hide", self)
        self.toggle_button.clicked.connect(self.toggle_visibility)

        self.fullscreen_button = QtWidgets.QPushButton("🖥 Fullscreen", self)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        self.exit_fullscreen_button = QtWidgets.QPushButton("🔳 Windowed", self)
        self.exit_fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.exit_fullscreen_button.hide()

        self.close_button = QtWidgets.QPushButton("❌ Close", self)
        self.close_button.clicked.connect(self.close)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.toggle_button)
        button_layout.addStretch()
        button_layout.addWidget(self.fullscreen_button)
        button_layout.addWidget(self.exit_fullscreen_button)
        button_layout.addWidget(self.close_button)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addLayout(button_layout)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        container = QtWidgets.QWidget()
        container.setLayout(self.main_layout)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(container)
        self.setLayout(layout)

        self.default_size = QtCore.QSize(320, 240)
        self.video_label.setFixedSize(self.default_size)
        self.setFixedSize(self.default_size)
        self.timer.start(30)

        self.scroll_detector = GestureScroll()
        self.fullscreen = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.scroll_detector.process(frame)
            size = self.video_label.size()
            frame = cv2.resize(frame, (size.width(), size.height()))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def toggle_visibility(self):
        visible = not self.video_label.isVisible()
        self.video_label.setVisible(visible)
        self.toggle_button.setText("👁 Hide" if visible else "👁 Show")

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
            self.setFixedSize(self.default_size)
            self.video_label.setFixedSize(self.default_size)
            self.fullscreen_button.show()
            self.exit_fullscreen_button.hide()
        else:
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            width = screen.width() // 2
            height = int(width * 0.75)
            new_size = QtCore.QSize(width, height)
            self.showMaximized()
            self.setFixedSize(screen.width(), screen.height())
            self.video_label.setFixedSize(new_size)
            self.fullscreen_button.hide()
            self.exit_fullscreen_button.show()
        self.fullscreen = not self.fullscreen

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = CameraOverlay()
    screen = app.primaryScreen().availableGeometry()
    win.move(screen.width() - win.width() - 20, screen.height() - win.height() - 20)
    win.show()
    sys.exit(app.exec())
