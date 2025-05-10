# visualizer_callback.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model


class TrainingVisualizerCallback(Callback):
    def __init__(self, X_val, y_val, class_names=None, interval=1, feature_map_img=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        self.interval = interval
        self.feature_map_img = feature_map_img
        self.epoch = 0
        os.makedirs("visualizations", exist_ok=True)
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        logs = logs or {}
        for k in self.history.keys():
            self.history[k].append(logs.get(k))

        self.plot_training_progress()

        if self.epoch % self.interval == 0:
            self.evaluate_performance()
            if self.feature_map_img is not None:
                self.visualize_feature_maps()

    def plot_training_progress(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"visualizations/training_epoch_{self.epoch}.png")
        plt.close()

    def evaluate_performance(self):
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_val, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (Epoch {self.epoch})")
        plt.savefig(f"visualizations/conf_matrix_epoch_{self.epoch}.png")
        plt.close()

        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        with open(f"visualizations/classification_report_epoch_{self.epoch}.txt", "w") as f:
            f.write(report)

    def visualize_feature_maps(self, max_filters=8):
        conv_layers = [layer for layer in self.model.layers if 'conv' in layer.name]
        outputs = [layer.output for layer in conv_layers]
        activation_model = Model(inputs=self.model.input, outputs=outputs)
        activations = activation_model.predict(self.feature_map_img[np.newaxis, ...])

        for layer_name, activation in zip([l.name for l in conv_layers], activations):
            fig, axes = plt.subplots(1, max_filters, figsize=(max_filters * 1.5, 2))
            fig.suptitle(f'{layer_name} - Epoch {self.epoch}')
            for i in range(min(max_filters, activation.shape[-1])):
                axes[i].imshow(activation[0, :, :, i], cmap='viridis')
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(f"visualizations/{layer_name}_epoch{self.epoch}.png")
            plt.close()
