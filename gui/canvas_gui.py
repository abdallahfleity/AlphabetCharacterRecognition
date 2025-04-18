import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt
import os

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 800)
        self.canvas = QPixmap(self.size())
        self.canvas.fill(Qt.white)
        self.drawing = False
        self.last_point = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point is not None:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.canvas.fill(Qt.white)
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖊️ Draw a Character")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #f5f5f5;")

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Canvas widget (white background)
        self.canvas = Canvas()
        self.canvas.setStyleSheet("background-color: white;")
        main_layout.addWidget(self.canvas)

        # Button panel (dark gray background)
        button_panel = QWidget()
        button_panel.setStyleSheet("background-color: #333333;")
        button_panel.setFixedHeight(100)

        # Button layout
        button_layout = QHBoxLayout(button_panel)
        button_layout.setContentsMargins(20, 10, 20, 10)
        button_layout.setSpacing(20)

        # Buttons
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clear_button")
        self.clear_btn.clicked.connect(self.canvas.clear)
        self.clear_btn.setFixedSize(120, 50)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setObjectName("predict_button")
        self.predict_btn.setFixedSize(120, 50)

        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.predict_btn)

        main_layout.addWidget(button_panel)

        # Load style file
        style_path = os.path.join(os.path.dirname(__file__), "style.qss")
        if os.path.exists(style_path):
            with open(style_path, "r") as style_file:
                self.setStyleSheet(self.styleSheet() + style_file.read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())