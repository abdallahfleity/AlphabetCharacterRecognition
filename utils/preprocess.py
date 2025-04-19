import idx2numpy
import numpy as np
import os
from PIL import Image

def resize_images_to_64x64(images):
    resized = []
    for img in images:
        # Convert single-channel 28x28 to PIL image
        pil_img = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
        pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)
        resized.append(np.array(pil_img) / 255.0)
    resized = np.array(resized)
    return np.expand_dims(resized, axis=-1)

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

    # ✅ Remove digits (labels 0–9) and shift labels down by 10
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    train_mask = train_labels >= 10
    test_mask = test_labels >= 10

    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask] - 10
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask] - 10

    # ✅ Resize to 64x64
    train_images = resize_images_to_64x64(train_images)
    test_images = resize_images_to_64x64(test_images)

    print(f"[INFO] Loaded and resized: {train_images.shape[0]} train, {test_images.shape[0]} test.")
    return (train_images, train_labels), (test_images, test_labels)
