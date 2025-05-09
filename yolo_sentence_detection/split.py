import os
import shutil
import random

def split_dataset(images_dir, labels_dir, output_dir, split_ratio=0.8):
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for split, file_list in [('train', train_images), ('val', val_images)]:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, folder, split), exist_ok=True)

        for img_file in file_list:
            txt_file = img_file.replace('.jpg', '.txt')
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(output_dir, 'images', split, img_file))
            shutil.copy(os.path.join(labels_dir, txt_file), os.path.join(output_dir, 'labels', split, txt_file))

    print(f"✅ Split {len(train_images)} training and {len(val_images)} validation images.")

split_dataset(
    images_dir='dataset/images',
    labels_dir='dataset/labels',
    output_dir='dataset',
    split_ratio=0.8
)
