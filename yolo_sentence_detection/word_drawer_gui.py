import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint, QDateTime

# Output directory
SAVE_DIR = "dataset/images"
os.makedirs(SAVE_DIR, exist_ok=True)

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 300)  # Wider for full words
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_image(self):
        return self.image

    def save_image(self):
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
        save_path = os.path.join(SAVE_DIR, f"word_{timestamp}.jpg")
        self.image.save(save_path)
        print(f"🖼️ Saved: {save_path}")

class WordDrawer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📝 Draw a Word (e.g. I LOVE YOU)")
        self.setGeometry(200, 200, 850, 400)

        layout = QVBoxLayout()
        self.canvas = Canvas()
        layout.addWidget(self.canvas)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Word Image")
        self.clear_button = QPushButton("Clear")
        self.status = QLabel("✏️ Draw a word and click save.")

        self.save_button.clicked.connect(self.save)
        self.clear_button.clicked.connect(self.canvas.clear)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.status)

        self.setLayout(layout)

    def save(self):
        self.canvas.save_image()
        self.status.setText("✅ Word image saved. Draw another or exit.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WordDrawer()
    window.show()
    sys.exit(app.exec_())
