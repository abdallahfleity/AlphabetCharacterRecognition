import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from utils.preprocess import load_emnist_byclass

# === 1. Load your saved model ===
model = load_model("model/emnist_byclass_augmented_complex_version7.keras")  # change path if needed

# === 2. Load EMNIST data ===
(_, _), (X_test, y_test) = load_emnist_byclass()

# === 3. One-hot encode labels (if needed for evaluation) ===
y_test_cat = to_categorical(y_test, num_classes=52)

# === 4. Make predictions ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# === 5. Generate confusion matrix and classification report ===
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=2))

# === 6. Plot confusion matrix ===
plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix, annot=False, cmap="Blues", fmt="g")
plt.title("Confusion Matrix - Version 7 (No Rotation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()