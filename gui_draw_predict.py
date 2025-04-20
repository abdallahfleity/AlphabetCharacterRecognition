import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load your trained model (Version 7 with handwritten data)
model = load_model("model/alphabet_cnn.keras")
print("✅ Loaded Version 7 Model!")

# EMNIST ByClass (letters only): 0 → 'A' ... 25 → 'Z', 26 → 'a' ... 51 → 'z'
emnist_labels = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.white)
        self.last_point = QPoint()
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.pixmap)
            pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.pixmap.fill(Qt.white)
        self.update()

    def get_image(self):
        return self.pixmap.toImage()

class DrawApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draw a Character - Version 7")
        self.canvas = Canvas()

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear)

        self.result_label = QLabel("✏️ Draw a letter and click Predict.")
        self.result_label.setStyleSheet("font-size: 18px; color: #333;")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.result_label)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.clear_btn)
        self.setLayout(layout)
        self.setFixedSize(300, 400)

    def preprocess_image(self, image):
        # Convert to grayscale
        gray_img = image.convertToFormat(QImage.Format_Grayscale8)
        ptr = gray_img.bits()
        ptr.setsize(gray_img.byteCount())
        arr = np.array(ptr).reshape((gray_img.height(), gray_img.width()))

        # Invert
        img = Image.fromarray(arr)
        img = Image.eval(img, lambda x: 255 - x)

        # Resize to 28x28 with no aspect ratio or padding
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Normalize and reshape
        img_np = np.array(img).astype("float32") / 255.0
        img_np = np.expand_dims(img_np, axis=(0, -1))

        return img_np

    def predict(self):
        image = self.canvas.get_image()
        processed_input = self.preprocess_image(image)

        prediction = model.predict(processed_input)[0]
        predicted_index = np.argmax(prediction)
        predicted_char = emnist_labels[predicted_index]
        confidence = prediction[predicted_index] * 100

        self.result_label.setText(f"🎯 Predicted: '{predicted_char}' ({confidence:.2f}%)")

        # Optional: visualize the preprocessed image
        import matplotlib.pyplot as plt
        plt.imshow(processed_input.squeeze(), cmap='gray')
        plt.title(f"Input for Prediction: '{predicted_char}'")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawApp()
    window.show()
    sys.exit(app.exec_())
