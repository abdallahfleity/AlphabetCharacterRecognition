import  os
from model.cnn_model import build_deep_cnn_model
from utils.preprocess import load_emnist_byclass
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Step 1: Load Data
def load_handwritten_dataset(path='data/handwritten_dataset', image_size=(28, 28)):
    X, y = [], []
    for label in os.listdir(path):
        folder_path = os.path.join(path, label)
        if os.path.isdir(folder_path) and len(label) == 1 and label.isalpha():
            label_index = ord(label) - 65 if label.isupper() else ord(label) - 71  # A-Z → 0–25, a-z → 26–51
            for img_file in os.listdir(folder_path):
                if img_file.endswith(".png"):
                    img = load_img(os.path.join(folder_path, img_file), color_mode='grayscale', target_size=image_size)
                    arr = img_to_array(img) / 255.0
                    X.append(arr)
                    y.append(label_index)
    return np.array(X), np.array(y)

custom_x, custom_y = load_handwritten_dataset()
print(f"[✅] Loaded handwritten dataset: {custom_x.shape[0]} samples")

# ===============================
# Step 2: Load EMNIST
# ===============================
(train_x, train_y), (test_x, test_y) = load_emnist_byclass()

# Merge custom samples into EMNIST
train_x = np.concatenate([train_x, custom_x])
train_y = np.concatenate([train_y, custom_y])

# Step 2: Combine and Split into train/val/test (70/20/10)
X_all = np.concatenate((train_x, test_x))
y_all = np.concatenate((train_y, test_y))

X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)

# Step 3: One-hot encode
y_train_cat = to_categorical(y_train, num_classes=52)
y_val_cat = to_categorical(y_val, num_classes=52)
y_test_cat = to_categorical(y_test, num_classes=52)

# Step 4: Data Augmentation
datagen = ImageDataGenerator(
   # rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Step 5: Build & Compile Model
model = build_deep_cnn_model()
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Train
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',        # You can also monitor 'val_accuracy'
    factor=0.5,                # Reduce learning rate by half
    patience=3,                # Wait 3 epochs of no improvement
    min_lr=1e-6,               # Don't go lower than this
    verbose=1                 # Show when it updates
)

model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    validation_data=(X_val, y_val_cat),
    epochs=25,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Step 7: Evaluate
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")

# Step 8: Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Save Model
model.save("model/emnist_byclass_augmented_complex_version5.h5")
print("📁 Model saved to model/emnist_byclass_augmented_complex_version7.keras")


