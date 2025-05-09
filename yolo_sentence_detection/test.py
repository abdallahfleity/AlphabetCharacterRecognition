import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from ultralytics import YOLO


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 300)
        self.setStyleSheet("background-color: white;")
        self.image = QPixmap(self.size())
        self.image.fill(Qt.white)
        self.last_point = QPoint()
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def save(self, filename):
        self.image.save(filename)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draw and Detect Letters with YOLOv8")
        self.canvas = Canvas()
        self.save_button = QPushButton("Save and Predict")
        self.clear_button = QPushButton("Clear")
        self.result_label = QLabel("Result will appear below")

        self.save_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.canvas.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.save_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def predict(self):
        save_path = "char_crops/drawn_input.jpg"
        self.canvas.save(save_path)

        try:
            model = YOLO("models/best_char_detector.pt")
        except Exception as e:
            self.result_label.setText(f"❌ Error loading model: {e}")
            return

        try:
            results = model.predict(source=save_path, imgsz=640, conf=0.25, save=False)
        except Exception as e:
            self.result_label.setText(f"❌ Error during prediction: {e}")
            return

        try:
            image = cv2.imread(save_path)
            if image is None:
                self.result_label.setText("❌ Could not load image for drawing boxes.")
                return

            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = results[0].names[int(cls)]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_qt_image = qt_image.scaled(600, 200, Qt.KeepAspectRatio)
            self.result_label.setPixmap(QPixmap.fromImage(scaled_qt_image))

        except Exception as e:
            self.result_label.setText(f"❌ Error displaying result on canvas: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
