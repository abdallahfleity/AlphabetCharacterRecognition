import idx2numpy
import numpy as np
import os

def load_emnist_byclass(path='data/gzip'):
    print("[INFO] Loading EMNIST ByClass (uppercase, lowercase only)...")

    train_images = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-train-labels-idx1-ubyte"))
    test_images = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-test-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-test-labels-idx1-ubyte"))

    # Normalize images
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Remove digits and shift labels
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    train_mask = train_labels >= 10
    test_mask = test_labels >= 10

    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask] - 10
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask] - 10

    # Reshape to match input shape
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    print(f"[INFO] Loaded EMNIST: {train_images.shape[0]} train, {test_images.shape[0]} test samples.")
    return (train_images, train_labels), (test_images, test_labels)
