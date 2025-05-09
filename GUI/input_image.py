
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap




class InputImage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Input Prompt")
        self.image_label = QLabel("No image loaded")
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(300, 300)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def prompt_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", "PNG Images (*.png);;jpeg images (*.jpg)"
        )
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(pixmap.size())
            return file_path
        return None