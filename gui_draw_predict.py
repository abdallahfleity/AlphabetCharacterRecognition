import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QPoint, QSize
from fastjsonschema.indent import indent
from numpy import character

from GUI.button_factory import ButtonFactory
from tensorflow.keras.models import load_model
from GUI.input_image import InputImage
from GUI.canvas_processing import CanvasProcessing
from GUI.API.google_ocr_api import APIGoogle

from GUI.tesseract_splitter import TesseractSplitter
# Load your trained model
model = load_model("emnist_data/best_emnist_letters.keras")

# EMNIST ByClass label map: A-Z + a-z
label_map = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]



def preprocess_canvas_image(image: QImage):
    if image.isNull():
        print("❌ Error: QImage is null.")
        return None

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
        screen_rect = QApplication.primaryScreen().availableGeometry()
        self.setFixedSize(screen_rect.width(), int(0.75*screen_rect.height()) )
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
        self.setWindowTitle("✍ Draw a Character or Sentence - EMNIST Predictor")
        self.model = model
        self.label_map = label_map  # same for label_map
        self.tesseract_splitter = TesseractSplitter()

        # self.setGeometry(300, 100, 800, 800)

        screen_rect = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen_rect)
        self.api_google = APIGoogle("C:/Users/96181/Downloads/ocrapi-459611-1cca03a9e183.json")


        layout = QVBoxLayout()
        self.canvas = Canvas()
        layout.addWidget(self.canvas)
        self.input_window = None



        button_layout = QHBoxLayout()
        self.predict_sentence_button = ButtonFactory.create_button("Predict_Sentence", bg_color='#010101')
        self.input_image_button = ButtonFactory.create_button("Input_Image", icon_path="GUI/GUI_Icons/Upload.png", bg_color='#010101')
        self.predict_button = ButtonFactory.create_button("Predict",bg_color='#010101',font_color='#3D65DB')
        self.clear_button = ButtonFactory.create_button("Clear",bg_color= '#010101')

        self.result_label = QLabel("Draw a letter and click Predict")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.result_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.result_label.setStyleSheet("color: #3D65DB; padding: 10px;")

        self.predict_sentence_button.clicked.connect(self.predict_sentence)
        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.canvas.clear)
        self.input_image_button.clicked.connect(self.show_input_image)

        button_layout.addWidget(self.predict_sentence_button)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.input_image_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def show_input_image(self):
        self.input_window = InputImage()
        path=self.input_window.prompt_image()
        if path:
            self.predict_input(path)
        else:
            print("No image selected")

    def predict_input(self,path):
        image = QImage(path)
        image_preprocessed = preprocess_canvas_image(image)
        if image_preprocessed is None:
            self.result_label.setText(" Could not  predict.")
            return
        prediction = model.predict(image_preprocessed)
        index=np.argmax(prediction)
        confidence=float(np.max(prediction)*100)
        character=label_map[index]
        self.result_label.setText(f" Prediction: {character} ({confidence:.2f}%)")

    def predict(self):
        img = self.canvas.get_image()
        preprocessed = preprocess_canvas_image(img)

        if preprocessed is None:
            self.result_label.setText(" Could not  drawing.")
            return

        prediction = model.predict(preprocessed)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        character = label_map[index]

        self.result_label.setText(f" Prediction: {character} ({confidence:.2f}%)")

    # here used for processing a full sentence

    def predict_sentence(self):
        qimage = self.canvas.get_image()
        saved_image = QImage(self.canvas.get_image())
        print(" QImage successfully retrieved from canvas.")
        if qimage.isNull():
            self.result_label.setText(" Invalid canvas image.")
            return

        try:
            word_boxes = self.api_google.get_word_boxes(qimage)
            print(f" Google OCR returned {len(word_boxes)} word boxes.")

            if not word_boxes:
                self.result_label.setText(" No words detected.")
                return

            painter = QPainter(qimage)
            painter.setPen(QPen(Qt.green, 2))
            for item in word_boxes:
                points = [QPoint(x, y) for x, y in item['bbox']]
                painter.drawPolygon(*points)
            painter.end()

            self.canvas.image = qimage
            self.canvas.update()
            self.result_label.setText(" Google OCR boxed the words.")
            self.test_word_cropping(saved_image)
        except Exception as e:
            print(" Google Vision OCR failed:", e)
            self.result_label.setText(" Failed to detect words.")

    def test_word_cropping(self, qimage: QImage):
        from PyQt5.QtGui import QImageWriter
        import os

        word_boxes = self.api_google.get_word_boxes(qimage)
        print(f"Detected {len(word_boxes)} words.")

        cropped_images = APIGoogle.crop_words_with_padding(qimage, word_boxes)

        os.makedirs("cropped_words", exist_ok=True)
        for idx, img in enumerate(cropped_images):
            path = f"cropped_words/word_{idx}.png"
            QImageWriter(path).write(img)
            print(f" Saved: {path}")

        all_char_crops = self.api_google.get_char_crops_from_words(cropped_images)

        # ✨ Predict the sentence using the character grid
        sentence = self.predict_extracted_sentence(all_char_crops)
        print("Predicted Sentence:", sentence)
        self.result_label.setText(f" Sentence: {sentence}")

        # Optionally save character images
        self.save_chars(all_char_crops)

    def save_chars(self, char_grid):
        import os
        from PyQt5.QtGui import QImageWriter

        os.makedirs("char_crops", exist_ok=True)
        for i, word_chars in enumerate(char_grid):
            for j, img in enumerate(word_chars):
                path = f"char_crops/word_{i}_char_{j}.png"
                QImageWriter(path).write(img)

    def predict_extracted_sentence(self, char_grid):
        # char by char prediction
        predicted_sentence = []

        for word_index, char_list in enumerate(char_grid):
            word = ""
            for char_index, char_qimg in enumerate(char_list):
                preprocessed = preprocess_canvas_image(char_qimg)
                if preprocessed is None:
                    print(f"⚠️ Skipping empty char at word {word_index}, index {char_index}")
                    continue
                prediction = self.model.predict(preprocessed)
                label_index = np.argmax(prediction)
                character = self.label_map[label_index]
                word += character
            predicted_sentence.append(word)

        return " ".join(predicted_sentence)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Predictor()
    window.show()
    sys.exit(app.exec_())