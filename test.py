

# Mute tensorflow debugging information console
# Mute tensorflow debugging information console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D,
                          Dropout, GlobalAveragePooling2D, Dense)
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x


def build_resnet(input_shape, nb_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def rotate(img):
    return np.rot90(np.fliplr(img))


def load_data(mat_file_path, width=28, height=28, max_=None):
    mat = loadmat(mat_file_path)
    mapping_raw = mat['dataset'][0][0][2]
    mapping = {kv[0]: kv[1:][0] for kv in mapping_raw}
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    train_imgs = mat['dataset'][0][0][0][0][0][0]
    train_lbls = mat['dataset'][0][0][0][0][0][1]
    test_imgs = mat['dataset'][0][0][1][0][0][0]
    test_lbls = mat['dataset'][0][0][1][0][0][1]

    if max_:
        train_imgs = train_imgs[:max_]
        train_lbls = train_lbls[:max_]
        test_imgs = test_imgs[:max_ // 6]
        test_lbls = test_lbls[:max_ // 6]

    # Filter for letters only
    allowed = [k for k, v in mapping.items() if 65 <= v <= 90 or 97 <= v <= 122]
    train_idx = [i for i, l in enumerate(train_lbls) if l in allowed]
    test_idx = [i for i, l in enumerate(test_lbls) if l in allowed]

    X = np.concatenate((train_imgs[train_idx], test_imgs[test_idx]), axis=0)
    y = np.concatenate((train_lbls[train_idx], test_lbls[test_idx]), axis=0)
    X = X.reshape(-1, height, width)
    X = np.array([rotate(img) for img in X])
    X = X.reshape(-1, height, width, 1).astype('float32') / 255.0

    # Re-map labels to 0...n
    label_vals = sorted(list(set(y.flatten())))
    label_map = {val: i for i, val in enumerate(label_vals)}
    y = np.array([label_map[v] for v in y.flatten()])

    nb_classes = len(label_map)

    # Stratified split
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    return (x_train, y_train), (x_val, y_val), mapping, nb_classes


def train(model, data, epochs=50, batch_size=256):
    (x_train, y_train), (x_val, y_val), _, nb_classes = data
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)

    datagen = ImageDataGenerator(
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    model_path = 'bin/best_emnist_letters.keras'
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              epochs=epochs,
              validation_data=(x_val, y_val),
              callbacks=[checkpoint])

    print("✔️ Training complete. Model saved as:", model_path)


if __name__ == '__main__':
    mat_file_path = "data/emnist-byclass.mat"
    width, height = 28, 28
    max_ = None  # Use full dataset
    epochs = 50

    if not os.path.exists('bin'):
        os.makedirs('bin')

    training_data = load_data(mat_file_path, width, height, max_)
    model = build_resnet((height, width, 1), training_data[3])
    train(model, training_data, epochs=epochs)
