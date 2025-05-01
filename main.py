import numpy as np
from model.cnn_model import build_deep_cnn_model
from utils.preprocess import load_emnist_byclass
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load EMNIST dataset
(train_x, train_y), (test_x, test_y) = load_emnist_byclass()

# Merge and split
X_all = np.concatenate((train_x, test_x))
y_all = np.concatenate((train_y, test_y))

X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp)

# One-hot encode
y_train_cat = to_categorical(y_train, num_classes=52)
y_val_cat = to_categorical(y_val, num_classes=52)
y_test_cat = to_categorical(y_test, num_classes=52)

# Data augmentation
datagen = ImageDataGenerator(
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5
)
datagen.fit(X_train)

# Build and compile model
model = build_deep_cnn_model()
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    validation_data=(X_val, y_val_cat),
    epochs=25,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate and save
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model.save("model/emnist_byclass_28x28_only.keras")
print("📁 Model saved to model/emnist_byclass_28x28_only.keras")
