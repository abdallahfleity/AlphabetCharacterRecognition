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