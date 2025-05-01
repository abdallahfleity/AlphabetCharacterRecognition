import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("emnist_data/best_emnist_letters.keras")

# EMNIST ByClass label map: A-Z + a-z
label_map = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

def preprocess_canvas_image(image: QImage):
    buffer = image.bits().asstring(image.byteCount())
    img_np = np.frombuffer(buffer, dtype=np.uint8).reshape((image.height(), image.width(), 4))
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
    gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]

    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    canvas = canvas.astype("float32") / 255.0
    canvas = np.expand_dims(canvas, axis=-1)
    canvas = np.expand_dims(canvas, axis=0)
    return canvas

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGBA8888)
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

class Predictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("✍️ Draw a Character - EMNIST Predictor")
        self.setGeometry(300, 100, 400, 400)

        layout = QVBoxLayout()
        self.canvas = Canvas()
        layout.addWidget(self.canvas)

        button_layout = QHBoxLayout()
        self.predict_button = QPushButton("✅ Predict")
        self.clear_button = QPushButton("🧹 Clear")
        self.result_label = QLabel("Draw a letter and click Predict")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.canvas.clear)

        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict(self):
        qimg = self.canvas.get_image()
        preprocessed = preprocess_canvas_image(qimg)
        if preprocessed is None:
            self.result_label.setText("❌ Could not process drawing.")
            return

        prediction = model.predict(preprocessed)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        character = label_map[index]

        self.result_label.setText(f"🔤 Prediction: {character} ({confidence:.2f}%)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Predictor()
    window.show()
    sys.exit(app.exec_())