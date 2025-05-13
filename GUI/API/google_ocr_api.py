import os
import io
from PyQt5.QtCore import QBuffer
from google.cloud import vision
from PyQt5.QtGui import QImage
import numpy as np
import cv2
from typing import List
import json
from google.api_core.exceptions import GoogleAPIError

class APIGoogle:
    def __init__(self, credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = vision.ImageAnnotatorClient()

    def qimage_to_bytes(self, qimage: QImage) -> bytes:
        from PyQt5.QtCore import QBuffer, QIODevice

        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qimage.save(buffer, "PNG")
        qbyte_array = buffer.data()
        buffer.close()

        return bytes(qbyte_array)

    def get_word_boxes(self, qimage: QImage):
        image_bytes = self.qimage_to_bytes(qimage)
        image = vision.Image(content=image_bytes)

        response = self.client.document_text_detection(image=image)
        word_boxes = []

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        text = ''.join([s.text for s in word.symbols])
                        bbox = [(v.x, v.y) for v in word.bounding_box.vertices]
                        word_boxes.append({'word': text, 'bbox': bbox})
        return word_boxes

    @staticmethod
    def crop_words_with_padding(qimage: QImage, word_boxes, top_padding=20, bottom_padding=20):

        # Convert QImage to NumPy array (RGBA)
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        img_np = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))

        cropped_images = []

        for word in word_boxes:
            bbox = word['bbox']  # List of 4 points: (x, y)
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]

            x_min = max(min(xs), 0)
            x_max = min(max(xs), width)
            y_min = max(min(ys) - top_padding, 0)
            y_max = min(max(ys) + bottom_padding, height)

            # Crop from the image
            cropped_np = img_np[y_min:y_max, x_min:x_max]

            # Convert back to QImage
            h, w, ch = cropped_np.shape
            bytes_per_line = ch * w
            cropped_qimage = QImage(cropped_np.tobytes(), w, h, bytes_per_line, QImage.Format_RGBA8888).copy()

            cropped_images.append(cropped_qimage)

        return cropped_images

    def get_char_crops_from_words(self, word_qimages: List[QImage], padding=5):
        all_char_crops = []

        for idx, word_img in enumerate(word_qimages):
            print(f"\n🔍 Processing word image {idx + 1}/{len(word_qimages)}")
            char_crops = []

            try:
                image_bytes = self.qimage_to_bytes(word_img)
                image = vision.Image(content=image_bytes)
                response = self.client.document_text_detection(image=image)

                full_text = response.full_text_annotation.text.strip()
                print(f"📄 Full OCR Text for word {idx + 1}: '{full_text}'")
                print(
                    f"🔠 Characters found: {sum(len(word.symbols) for para in response.full_text_annotation.pages[0].blocks[0].paragraphs for word in para.words)}")

                qimg_copy = word_img.convertToFormat(QImage.Format_RGBA8888)
                w, h = qimg_copy.width(), qimg_copy.height()
                ptr = qimg_copy.bits()
                ptr.setsize(qimg_copy.sizeInBytes())
                img_np = np.array(ptr, dtype=np.uint8).reshape((h, w, 4))

                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        for para in block.paragraphs:
                            for word in para.words:
                                for symbol in word.symbols:
                                    symbol_text = symbol.text
                                    bbox = [(v.x, v.y) for v in symbol.bounding_box.vertices]
                                    print(f"🔡 Symbol: '{symbol_text}' | Box: {bbox}")

                                    xs = [p[0] for p in bbox]
                                    ys = [p[1] for p in bbox]

                                    x_min = max(min(xs) - padding, 0)
                                    x_max = min(max(xs) + padding, w)
                                    y_min = max(min(ys) - padding, 0)
                                    y_max = min(max(ys) + padding, h)

                                    char_crop_np = img_np[y_min:y_max, x_min:x_max]
                                    ch, cw, cch = char_crop_np.shape
                                    bytes_per_line = cw * cch
                                    char_qimage = QImage(char_crop_np.tobytes(), cw, ch, bytes_per_line,
                                                         QImage.Format_RGBA8888).copy()
                                    char_crops.append(char_qimage)

            except GoogleAPIError as api_err:
                print(f"❌ Google Vision API error on word {idx + 1}: {api_err}")
            except Exception as e:
                print(f"❌ General error on word {idx + 1}: {e}")

            all_char_crops.append(char_crops)

        print(f"\n✅ Done. Total words processed: {len(all_char_crops)}")
        return all_char_crops