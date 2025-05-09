import numpy as np
import cv2
from PyQt5.QtGui import QImage

class CanvasProcessing:
    def qimage_to_np(image: QImage):
        if image.isNull():
            raise ValueError("❌ QImage is null!")

        # 🔁 Force format to RGBA8888 before accessing bits
        image = image.convertToFormat(QImage.Format_RGBA8888)

        width, height = image.width(), image.height()
        buffer = image.bits()
        buffer.setsize(image.byteCount())
        img_np = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)

    @staticmethod
    def crop_and_resize(gray_img, padding=10):

        coords = cv2.findNonZero(gray_img)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = gray_img[y:y+h, x:x+w]
        padded = cv2.copyMakeBorder(cropped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
        return padded

    @staticmethod
    def np_to_qimage(gray_img):

        height, width = gray_img.shape
        qimg = QImage(gray_img.data, width, height, width, QImage.Format_Grayscale8)
        return qimg.copy()

    @staticmethod
    def process_canvas(qimage: QImage):

        gray = CanvasProcessing.qimage_to_np(qimage)
        gray = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = sorted(boxes, key=lambda b: b[0])  # sort left to right

        word_images = []
        current_word = []
        prev_x = None
        prev_w = None

        for (x, y, w, h) in boxes:
            if prev_x is not None:
                gap = x - (prev_x + prev_w)
                if gap > 20:  # Adjust threshold for word spacing
                    if current_word:
                        word_img = CanvasProcessing.extract_word_image(binary, current_word)
                        word_images.append(CanvasProcessing.np_to_qimage(word_img))
                        current_word = []
            current_word.append((x, y, w, h))
            prev_x, prev_w = x, w

        if current_word:
            word_img = CanvasProcessing.extract_word_image(binary, current_word)
            word_images.append(CanvasProcessing.np_to_qimage(word_img))

        return word_images

    @staticmethod
    def extract_word_image(binary_img, boxes):

        x_coords = [x for (x, y, w, h) in boxes]
        y_coords = [y for (x, y, w, h) in boxes]
        w_coords = [w for (x, y, w, h) in boxes]
        h_coords = [h for (x, y, w, h) in boxes]

        x_min = min(x_coords)
        x_max = max([x + w for x, w in zip(x_coords, w_coords)])
        y_min = min(y_coords)
        y_max = max([y + h for y, h in zip(y_coords, h_coords)])

        word_crop = binary_img[y_min:y_max, x_min:x_max]
        return CanvasProcessing.crop_and_resize(word_crop)

    @staticmethod
    def process_word(qimage: QImage):
        if qimage.isNull():
            raise ValueError("❌ QImage passed to process_word is null.")

        try:
            print("🧪 process_word: Converting QImage to NumPy")
            gray = CanvasProcessing.qimage_to_np(qimage)
            print("✅ Grayscale conversion done")

            gray = cv2.bitwise_not(gray)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print("✅ Thresholding done")

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours]
            boxes = sorted(boxes, key=lambda b: b[0])
            print(f"🔍 Found {len(boxes)} characters")

            characters = []
            for idx, (x, y, w, h) in enumerate(boxes):
                if w < 5 or h < 5:
                    print(f"⚠️ Skipping noise box at idx={idx + 1} (w={w}, h={h})")
                    continue
                print(f"🔹 Char {idx + 1}: x={x}, y={y}, w={w}, h={h}")
                char_crop = binary[y:y + h, x:x + w]

                if char_crop.size == 0:
                    print(f"⚠️ Skipping empty crop at index {idx}")
                    continue

                try:
                    char_crop = CanvasProcessing.crop_and_resize(char_crop, padding=5)
                    char_qimage = CanvasProcessing.np_to_qimage(char_crop)
                    characters.append(char_qimage)
                except Exception as e:
                    print(f"❌ Error processing character {idx + 1}: {e}")
                    continue

            print("✅ process_word finished")
            return characters

        except Exception as e:
            print("❌ process_word failed:", e)
            return []

    @staticmethod
    def process_sentence(qimage: QImage):
        print("start processing sentence:")
        all_characters = []
        word_images = CanvasProcessing.process_canvas(qimage)
        i=0
        for word_img in word_images:
            i=i+1
            print(f"processing word {i} of {len(word_images)}")
            chars = CanvasProcessing.process_word(word_img)
            all_characters.append(chars)

        return all_characters