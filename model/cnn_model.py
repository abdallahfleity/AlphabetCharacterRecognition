from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    Add, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, SpatialDropout2D
)

def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_deep_cnn_model():
    inputs = Input(shape=(28, 28, 1))

    # Initial lightweight conv to enhance low-level features
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)  # Added one more deep block

    # Global feature compression
    x = SpatialDropout2D(0.3)(x)
    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(52, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
