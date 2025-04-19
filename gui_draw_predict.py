import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model
from PIL import Image

# === Load your trained model (Version 6) ===
model = load_model("model/emnist_byclass_augmented_complex_version7.keras")
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
        from scipy.ndimage import center_of_mass
        import cv2

        # Step 1: Convert to grayscale NumPy array
        image = self.canvas.get_image().convertToFormat(QImage.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape((image.height(), image.width()))

        # Step 2: Invert colors (black on white)
        img = Image.fromarray(arr)
        img = Image.eval(img, lambda x: 255 - x)
        img_np = np.array(img)

        # Step 3: Auto-crop non-empty regions
        coords = cv2.findNonZero(255 - img_np)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_np[y:y + h, x:x + w]

        # Step 4: Resize with padding to 28x28
        target_size = 28
        h, w = cropped.shape
        scale = target_size / max(h, w)
        resized = cv2.resize(cropped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        pad_top = (target_size - resized.shape[0]) // 2
        pad_bottom = target_size - resized.shape[0] - pad_top
        pad_left = (target_size - resized.shape[1]) // 2
        pad_right = target_size - resized.shape[1] - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # Step 5: Deskew using moments
        moments = cv2.moments(padded)
        if abs(moments["mu02"]) > 1e-2:
            skew = moments["mu11"] / moments["mu02"]
            M = np.float32([[1, skew, -0.5 * skew * padded.shape[0]], [0, 1, 0]])
            padded = cv2.warpAffine(padded, M, (target_size, target_size),
                                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        # Step 6: Binary thresholding to sharpen
        _, binarized = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 7: Normalize & reshape
        binarized = binarized.astype("float32") / 255.0
        binarized = np.expand_dims(binarized, axis=(0, -1))

        # Step 8: Visualize (optional)
        import matplotlib.pyplot as plt
        plt.imshow(binarized.squeeze(), cmap='gray')
        plt.title("Processed Input")
        plt.axis('off')
        plt.show()

        # Step 9: Predict
        pred = model.predict(binarized)[0]
        index = np.argmax(pred)
        confidence = pred[index] * 100
        predicted_char = emnist_labels[index]

        self.result_label.setText(f"🎯 Predicted: '{predicted_char}' ({confidence:.2f}%)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawApp()
    window.show()
    sys.exit(app.exec_())
