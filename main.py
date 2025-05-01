import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.cnn_model import build_emnist_cnn
from utils.preprocess import load_emnist_byclass_letters_only

# Load EMNIST (letters only)
(train_x, train_y), (test_x, test_y) = load_emnist_byclass_letters_only()

# Combine train/test then split into train/val/test
X_all = np.concatenate([train_x, test_x])
y_all = np.concatenate([train_y, test_y])
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

# One-hot encode
y_train_cat = to_categorical(y_train, 52)
y_val_cat = to_categorical(y_val, 52)
y_test_cat = to_categorical(y_test, 52)

# Data augmentation
datagen = ImageDataGenerator(
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10
)
datagen.fit(X_train)

# Build model
model = build_emnist_cnn()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
earlystop = EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("model/best_emnist_model.keras", save_best_only=True, monitor='val_accuracy', mode='max')

# Train
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    validation_data=(X_val, y_val_cat),
    epochs=30,
    callbacks=[earlystop, checkpoint]
)

# Plot accuracy and loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()
plt.show()

# Evaluation
model.load_weights("model/best_emnist_model.keras")
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
