import idx2numpy
import numpy as np
import os

def load_emnist_byclass_letters_only(path='data/gzip'):
    print("[INFO] Loading EMNIST ByClass (A-Z, a-z only)...")

    train_images = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-train-images-idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-train-labels-idx1-ubyte"))
    test_images = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-test-images-idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(os.path.join(path, "emnist-byclass-test-labels-idx1-ubyte"))

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    # Filter to keep only letters (labels >= 10), and shift them to 0–51
    train_mask = train_labels >= 10
    test_mask = test_labels >= 10
    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask] - 10
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask] - 10

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    print(f"[INFO] Loaded EMNIST (letters only): {train_images.shape[0]} train, {test_images.shape[0]} test")
    return (train_images, train_labels), (test_images, test_labels)
