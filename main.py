import os
import numpy as np
from model.cnn_model import build_deep_cnn_model
from utils.preprocess import load_emnist_byclass
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ===============================
# Step 1: Load Your Handwritten Dataset
# ===============================
def load_handwritten_dataset(path='data/handwritten_dataset', image_size=(64, 64)):
    X, y = [], []
    for label in os.listdir(path):
        folder_path = os.path.join(path, label)
        if not os.path.isdir(folder_path):
            continue

        # Accept folders like a_lower, B_upper
        if '_lower' in label or '_upper' in label:
            char = label[0]
            label_index = ord(char) - 65 if char.isupper() else ord(char) - 71  # A-Z → 0–25, a-z → 26–51
            for img_file in os.listdir(folder_path):
                if img_file.endswith(".png"):
                    img = load_img(os.path.join(folder_path, img_file), color_mode='grayscale', target_size=image_size)
                    arr = img_to_array(img) / 255.0
                    X.append(arr)
                    y.append(label_index)

    if len(X) == 0:
        print("[⚠️] No handwritten samples found.")
    return np.array(X), np.array(y)

custom_x, custom_y = load_handwritten_dataset()
print(f"[✅] Loaded handwritten dataset: {custom_x.shape[0]} samples")

# ===============================
# Step 2: Load EMNIST Dataset
# ===============================
(train_x, train_y), (test_x, test_y) = load_emnist_byclass(image_size=(64, 64))

# ===============================
# Step 3: Merge Your Data with EMNIST
# ===============================
if custom_x.shape[0] > 0:
    if len(custom_x.shape) == 3:
        custom_x = np.expand_dims(custom_x, -1)

    train_x = np.concatenate([train_x, custom_x])
    train_y = np.concatenate([train_y, custom_y])
    print("[✅] Handwritten samples merged into EMNIST.")
else:
    print("[⚠️] No handwritten samples merged. Continuing with EMNIST only.")

# ===============================
# Step 4: Split into Train/Val/Test
# ===============================
X_all = np.concatenate((train_x, test_x))
y_all = np.concatenate((train_y, test_y))

X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)

# ===============================
# Step 5: One-Hot Encode Labels
# ===============================
y_train_cat = to_categorical(y_train, num_classes=52)
y_val_cat = to_categorical(y_val, num_classes=52)
y_test_cat = to_categorical(y_test, num_classes=52)

# ===============================
# Step 6: Data Augmentation
# ===============================
datagen = ImageDataGenerator(
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5
)
datagen.fit(X_train)

# ===============================
# Step 7: Build Model
# ===============================
model = build_deep_cnn_model(input_shape=(64, 64, 1))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# Step 8: Train Model
# ===============================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    validation_data=(X_val, y_val_cat),
    epochs=25,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ===============================
# Step 9: Evaluate & Save
# ===============================
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
model.save("model/emnist_byclass_augmented_complex_version7_64x64.keras")
print("📁 Model saved to model/emnist_byclass_augmented_complex_version7_64x64.keras")
