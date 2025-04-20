
import idx2numpy
import numpy as np
import os

def load_emnist_byclass(path='data/gzip'):
    print("[INFO] Loading EMNIST ByClass (uppercase, lowercase only)...")

    # Load raw data
    train_images = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-train-labels-idx1-ubyte"))
    test_images = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-test-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-test-labels-idx1-ubyte"))

    # Normalize image data
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Expand dims to (28, 28, 1)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Flatten labels
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    # ✅ Remove digits (labels 0–9) and shift labels down by 10
    train_mask = train_labels >= 10
    test_mask = test_labels >= 10

    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask] - 10

    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask] - 10

    print(f"[INFO] Loaded: {train_images.shape[0]} train, {test_images.shape[0]} test.")
    return (train_images, train_labels), (test_images, test_labels)

