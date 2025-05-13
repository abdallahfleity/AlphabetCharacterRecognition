import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import numpy as np
import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPainter, QPen, QColor


class TesseractSplitter:
    def __init__(self):
        pass

    def qimage_to_opencv(self, qimage):
        """Convert QImage to OpenCV format (numpy array) safely."""
        try:
            qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
            print(" QImage successfully converted to Format_RGBA8888.")
        except Exception as e:
            print("", e)
            return None  # Don't call UI here!

        width = qimage.width()
        height = qimage.height()

        try:
            ptr = qimage.bits()
            ptr.setsize(qimage.sizeInBytes())
            arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()
            print(" QImage successfully converted to numpy array.")
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            print(" Failed during QImage to OpenCV conversion:", e)
            return None

    def split_words_from_cv(self, opencv_image):
        """Use Tesseract on an OpenCV image to get word bounding boxes."""
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

        custom_config = r'--oem 3 --psm 6'
        result = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)

        word_boxes = []
        for i in range(len(result['text'])):
            if int(result['conf'][i]) > 0 and result['text'][i].strip():
                x, y, w, h = result['left'][i], result['top'][i], result['width'][i], result['height'][i]
                word_boxes.append({'word': result['text'][i], 'bbox': (x, y, w, h)})
        return word_boxes

    def draw_word_boxes_cv(self, img, word_boxes):
        """Draw boxes on an OpenCV image."""
        for word in word_boxes:
            x, y, w, h = word['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, word['word'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return img

