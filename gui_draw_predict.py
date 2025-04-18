import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model
from PIL import Image

# === Load your trained model (Version 6) ===
model = load_model("model/emnist_byclass_augmented_complex_version6.h5")
print("✅ Loaded Version 6 Model!")

# EMNIST ByClass (letters only) mapping: 0 → 'A', ..., 25 → 'Z', 26 → 'a', ..., 51 → 'z'
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
        self.setWindowTitle("Draw a Character - Version 6")
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

    def predict(self):
        image = self.canvas.get_image().convertToFormat(QImage.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape((image.height(), image.width()))

        # Step 1: Convert to PIL and invert
        img = Image.fromarray(arr)
        img = Image.eval(img, lambda x: 255 - x)  # Invert (black on white)

        # Step 2: Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # Step 3: Center the character using center of mass
        from scipy.ndimage import center_of_mass
        cy, cx = center_of_mass(img_np > 30)  # Threshold to find ink
        shift_x, shift_y = int(14 - cx), int(14 - cy)
        img_np = np.roll(img_np, shift_y, axis=0)
        img_np = np.roll(img_np, shift_x, axis=1)

        # Step 4: Normalize and reshape
        img_np = img_np.astype("float32") / 255.0
        img_np = np.expand_dims(img_np, axis=(0, -1))  # Shape: (1, 28, 28, 1)

        # Step 5: Optional — Visualize what the model sees
        import matplotlib.pyplot as plt
        plt.imshow(img_np.squeeze(), cmap='gray')
        plt.title("What the model sees")
        plt.axis('off')
        plt.show()

        # Step 6: Predict
        pred = model.predict(img_np)[0]
        index = np.argmax(pred)
        confidence = pred[index] * 100
        predicted_char = emnist_labels[index]

        self.result_label.setText(f"🎯 Predicted: '{predicted_char}' ({confidence:.2f}%)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawApp()
    window.show()
    sys.exit(app.exec_())
