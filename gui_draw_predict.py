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
# Load your trained model
model = load_model("emnist_data/best_emnist_letters.keras")

# EMNIST ByClass label map: A-Z + a-z
label_map = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

def preprocess_canvas_image2(image: QImage):
    if image is None or image.isNull():
        print("❌ preprocess_canvas_image2: Null image passed.")
        return None

    try:
        # ✅ Force image to RGBA format
        image = image.convertToFormat(QImage.Format_RGBA8888)

        width, height = image.width(), image.height()
        buffer = image.bits()
        buffer.setsize(image.byteCount())

        img_np = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
        gray = cv2.bitwise_not(gray)
    except Exception as e:
        print("❌ Error converting image to NumPy:", e)
        return None

    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            print("⚠️ No non-zero pixels found in character.")
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w < 5 or h < 5:
            print(f"⚠️ Skipping too-small character: w={w}, h={h}")
            return None

        cropped = thresh[y:y+h, x:x+w]

        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        canvas = canvas.astype("float32") / 255.0
        canvas = np.expand_dims(canvas, axis=-1)
        canvas = np.expand_dims(canvas, axis=0)
        return canvas

    except Exception as e:
        print("❌ Error in preprocess_canvas_image2:", e)
        return None

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
        self.setFixedSize(screen_rect.width()//2, int(0.75*screen_rect.height()) )
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
        self.model = model  # model was loaded globally, now attached to this instance
        self.label_map = label_map  # same for label_map

        # self.setGeometry(300, 100, 800, 800)

        screen_rect = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen_rect)



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
            self.result_label.setText("❌ Could not  predict.")
            return
        prediction = model.predict(image_preprocessed)
        index=np.argmax(prediction)
        confidence=float(np.max(prediction)*100)
        character=label_map[index]
        self.result_label.setText(f"🔤 Prediction: {character} ({confidence:.2f}%)")

    def predict(self):
        img = self.canvas.get_image()
        preprocessed = preprocess_canvas_image(img)

        if preprocessed is None:
            self.result_label.setText("❌ Could not  drawing.")
            return

        prediction = model.predict(preprocessed)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        character = label_map[index]

        self.result_label.setText(f"🔤 Prediction: {character} ({confidence:.2f}%)")

    # here used for processing a full sentence

    def predict_sentence(self):
        img = self.canvas.get_image()
        all_words = CanvasProcessing.process_sentence(img)

        sentence_chars = []

        for word_index, word in enumerate(all_words):
            print(f"🔠 Predicting word {word_index + 1}/{len(all_words)}...")
            word_chars = []

            for char_index, char_img in enumerate(word):
                print(f"  ✏️ Character {char_index + 1} in word {word_index + 1}")
                preprocessed = preprocess_canvas_image2(char_img)

                if preprocessed is None:
                    print("    ⚠️ Skipping: preprocessing returned None")
                    word_chars.append("?")
                    continue

                if preprocessed.shape != (1, 28, 28, 1):
                    print(f"    ⚠️ Skipping: invalid shape {preprocessed.shape}")
                    word_chars.append("?")
                    continue

                try:
                    prediction = self.model.predict(preprocessed)
                    index = np.argmax(prediction)
                    character = self.label_map[index]
                    print(f"    ✅ Predicted: {character}")
                    word_chars.append(character)
                except Exception as e:
                    print(f"    ❌ Prediction failed: {e}")
                    word_chars.append("?")

            sentence_chars.append(word_chars)

        sentence_string = " ".join("".join(word) for word in sentence_chars)
        print("📝 Predicted sentence:", sentence_string)
        self.result_label.setText(f"📝 Predicted: {sentence_string}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Predictor()
    window.show()
    sys.exit(app.exec_())