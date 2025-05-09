# AlphabetCharacterRecognition

## 📥 How to Set Up the Dataset

1. Go to the official [EMNIST dataset page](https://www.nist.gov/itl/products-and-services/emnist-dataset).
2. Scroll down to the **"Where to download?"** section.
3. Click on **"matlab format as the original MNIST dataset"** and download the ZIP file.
4. Unzip the file and extract its contents into your project's `data` folder.
   - The `data` folder should include the `gzip` folder which is automatically added after unzipping the downloaded ZIP file.
5. Inside the `data` folder, locate the `gzip` subfolder.
6. Using WinRAR or any archive extraction tool, extract the contents of the `gzip` folder.
   - Tip: Right-click on the `data` folder, go to **Properties**, and **uncheck the "Read-only"** box to avoid permission issues (especially if your Windows user does not have root privileges).
7. Locate and extract the file:
   - `emnist-byclass.mat` (not `.train.mat`, the `.mat` file is already a full split containing both train and test sets.)
8. Place `emnist-byclass.mat` inside the `data` folder (or wherever your training script is configured to load from).

#### Your own dataset 

- **you can also without download the EMNIST you can draw your own dataset** 
- **At the end of this file you can find the code suitable to train your own dataset**

---

You can now proceed with training your model using this `.mat` file. Ensure you update your training script’s file path accordingly:

   ## 🧠 How the Model Was Trained (Canvas-Optimized)

This project uses a deep **ResNet-style Convolutional Neural Network** trained on the EMNIST ByClass dataset, specifically **filtered to only include alphabet characters** (`A–Z` and `a–z`).

> ⚠️ **Note:** The trained model (`best_emnist_letters.keras`) is optimized for **drawn characters on a canvas**. It is **not designed** for printed or scanned characters.

### 🔧 Model Features

- ✅ **Alphabet-only filtering** (52 classes)
- ✅ **Deep CNN with residual blocks (ResNet-inspired)**
- ✅ **Stratified split** (90% training / 10% validation)
- ✅ **Preprocessing:** rotation, normalization, grayscale
- ✅ **Data Augmentation:** zoom, width/height shift
- ✅ **Batch Normalization** and Dropout regularization
- ✅ **Early model saving using `ModelCheckpoint`**
- 💾 Model saved to: `bin/best_emnist_letters.keras`
- 💡 Label map saved to: `bin/mapping.p`

---

### 🏋️ Full Training Code (train_resnet_emnist.py)

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D,
                          Dropout, GlobalAveragePooling2D, Dense)
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def build_resnet(input_shape, nb_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def rotate(img):
    return np.rot90(np.fliplr(img))

def load_data(mat_file_path, width=28, height=28, max_=None):
    mat = loadmat(mat_file_path)
    mapping_raw = mat['dataset'][0][0][2]
    mapping = {kv[0]: kv[1:][0] for kv in mapping_raw}
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    train_imgs = mat['dataset'][0][0][0][0][0][0]
    train_lbls = mat['dataset'][0][0][0][0][0][1]
    test_imgs = mat['dataset'][0][0][1][0][0][0]
    test_lbls = mat['dataset'][0][0][1][0][0][1]

    if max_:
        train_imgs = train_imgs[:max_]
        train_lbls = train_lbls[:max_]
        test_imgs = test_imgs[:max_ // 6]
        test_lbls = test_lbls[:max_ // 6]

    allowed = [k for k, v in mapping.items() if 65 <= v <= 90 or 97 <= v <= 122]
    train_idx = [i for i, l in enumerate(train_lbls) if l in allowed]
    test_idx = [i for i, l in enumerate(test_lbls) if l in allowed]

    X = np.concatenate((train_imgs[train_idx], test_imgs[test_idx]), axis=0)
    y = np.concatenate((train_lbls[train_idx], test_lbls[test_idx]), axis=0)
    X = np.array([rotate(img) for img in X]).reshape(-1, height, width, 1).astype('float32') / 255.0

    label_vals = sorted(list(set(y.flatten())))
    label_map = {val: i for i, val in enumerate(label_vals)}
    y = np.array([label_map[v] for v in y.flatten()])
    nb_classes = len(label_map)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    return (x_train, y_train), (x_val, y_val), mapping, nb_classes

def train(model, data, epochs=50, batch_size=256):
    (x_train, y_train), (x_val, y_val), _, nb_classes = data
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)

    datagen = ImageDataGenerator(
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    model_path = 'bin/best_emnist_letters.keras'
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              epochs=epochs,
              validation_data=(x_val, y_val),
              callbacks=[checkpoint])

    print("✔️ Training complete. Model saved as:", model_path)

if __name__ == '__main__':
    mat_file_path = "emnist_data/emnist-byclass.mat"
    width, height = 28, 28
    max_ = None
    epochs = 50

    if not os.path.exists('bin'):
        os.makedirs('bin')

    training_data = load_data(mat_file_path, width, height, max_)
    model = build_resnet((height, width, 1), training_data[3])
    train(model, training_data, epochs=epochs)
```
```python gui predict canvas
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
 
```
### Your own dataset code 
> ⚠️ **Note:** be careful on how to set up your folders for your own dataset  
> data/
└── handwritten_dataset/
    ├── A_upper/
    ├── a_lower/
    ├── B_upper/
    ├── b_lower/
    ├── C_upper/
    ├── c_lower/
    └── ... (continue for all letters A–Z and a–z)
> data/handwritten_dataset/A_upper/
    A1.png
    A2.png
    ...



```python 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Load your handwritten dataset (assumed structure: a_lower/, B_upper/...)
def load_dataset(data_dir='data/handwritten_dataset_28', image_size=(28, 28)):
    X, y = [], []
    for folder in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, folder)
        if not os.path.isdir(label_path): continue

        char = folder[0]
        label = ord(char) - 65 if char.isupper() else ord(char) - 71  # A-Z → 0–25, a-z → 26–51

        for img_name in os.listdir(label_path):
            if not img_name.endswith(".png"): continue
            img_path = os.path.join(label_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=image_size)
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label)

    return np.array(X), np.array(y)

# === Load and prepare data
X, y = load_dataset()
print(f"✅ Loaded {len(X)} samples.")

X = X.reshape(-1, 28, 28, 1)
y_cat = to_categorical(y, num_classes=52)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42, stratify=y)

# === Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(52, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# === Train model
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1)

# === Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# === Predict and analyze
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# === Plot predictions
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    pred_chr = chr(y_pred[i]+65) if y_pred[i] < 26 else chr(y_pred[i]+71)
    true_chr = chr(y_true[i]+65) if y_true[i] < 26 else chr(y_true[i]+71)
    plt.title(f"P:{pred_chr}\nT:{true_chr}")
    plt.axis('off')
plt.tight_layout()
plt.show()
model.save("model/alphabet_cnn.keras")
print("📁 Model saved to 'model/alphabet_cnn.keras'")

```
