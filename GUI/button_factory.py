from PyQt5.QtWidgets import QPushButton
from scipy.signal import buttap
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QSize

class ButtonFactory:
    @staticmethod
    def create_button(text, width=-1, height=-1, bg_color='#3DD5DB',
                      font_family="Arial", font_size=12, icon_path=None, icon_size=(24, 24),font_color='#3D65DB'):
        button=QPushButton(text)
        if icon_path:
            button.setIcon(QIcon(icon_path))
            button.setIconSize(QSize(icon_size[0], icon_size[1]))
        if width>0 and height>0:
            button.setFixedSize(width, height)
            padding = ""
        else:
            padding= "4px 6px;"


        button.setStyleSheet(f"background-color: {bg_color};"
                             f"font-family: {font_family}; "
                             f"font-size: {font_size};"
                             f"color:{font_color}; "
                             f"padding:{padding}"
                             f"border-radius: 5px;")
        button.setFont(QFont(font_family, font_size))

        return button

