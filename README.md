# AlphabetCharacterRecognition
## 📥 How to Set Up the Dataset

1. Go to the official [EMNIST dataset page](https://www.nist.gov/itl/products-and-services/emnist-dataset).
2. Scroll down to the **"Where to download?"** section.
3. Click on **Binary format as the original MNIST dataset** and download the ZIP file.
4. Unzip the file and extract its contents into your project's `data` folder.
(data folder shou include gzip folder that unzipping downloaded folder automatically it will be added in data folder)
5. Inside the `data` folder, locate the `gzip` subfolder.
6. Using WinRAR or any extraction tool, extract the following 4 folders from it:
   (select the extract here option also it's better to go to data folder and right click select uncheck the read only box
    MY USER ABDALLAH IS NOT THE ROOT SO I ISSUED THIS ERROR   )
   - `emnist-byclass-train-images-idx3-ubyte`
   - `emnist-byclass-train-labels-idx1-ubyte`
   - `emnist-byclass-test-images-idx3-ubyte`
   - `emnist-byclass-test-labels-idx1-ubyte`
   

## 🤖 Using the Pretrained Model

You **do not need to retrain** the model — it's already included in the project under the `model/` directory as:

- `emnist_byclass_augmented_adam.h5`
- `emnist_byclass_augmented_rmsprop.h5`
## About versions

1. version7
- `versions 7.keras inlcude also callbacks in reduce when overfit occur valid decrease and accuracy increase  `
- `also include dataset emnist + handwritten`

## requirements file 

You have to follow and download all requirements from requirements.txt using teh appropriate command in terminal,**"Be carefully about versions"** 
- `be carefull with PyQt5 you have to download PyQt5-5.15.9 / PyQt5-5.15.2 to be compatible with GUI Designer tool`
- `to downloaded run the command in terminal:  pip install pyqt5-tools /  pip install pyqt5-tools==5.15.4.3.2 --force-reinstall `
- `After donloading him you will find it in as designer.exe : `
          **".venv\Lib\site-packages\qt5_applications\Qt\bin"**  or 
          **".venv\Lib\site-packages\pyqt5_tools"** 


## 🧠 If You Want to Retrain
- `emnist_byclass_augmented_complex_version4.h5`
this version is using only cnn using this code below in **"cnn_model.py"**



If you want to see about the training process, you can view the code used to train both models using **Adam** and **RMSprop** optimizers in:
```python
from model.cnn_model import build_cnn_model
from utils.preprocess import load_emnist_byclass
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the data
(train_x, train_y), (test_x, test_y) = load_emnist_byclass()

# Step 2: One-hot encode the labels (62 classes)
train_y_cat = to_categorical(train_y, num_classes=62)
test_y_cat = to_categorical(test_y, num_classes=62)

# Step 3: Build the CNN model
model = build_cnn_model()

# Step 4: Compile the model using RMSprop/Adam
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Summary
model.summary()

# Step 6: Set up early stopping
early_stop = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

# Step 7: Set up data augmentation
# It's used to help the model generalize better—like recognizing a slightly off-centered "A"
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(train_x)

# Step 8: Train the model using augmented data
model.fit(
    datagen.flow(train_x, train_y_cat, batch_size=128),
    validation_data=(test_x, test_y_cat),
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)

# Step 9: Evaluate final performance
loss, acc = model.evaluate(test_x, test_y_cat)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")

# Step 10: Save the trained model
model.save("model/emnist_byclass_augmented_adam.h5")
print("📁 Model saved to model/emnist_byclass_augmented_adam.h5")

---


