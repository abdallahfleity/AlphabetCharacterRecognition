from PyQt5.QtWidgets import QPushButton


class ButtonFactory:
    @staticmethod
    def create_button(text, width=100, height=40, bg_color="#3498db",
                      font_family="Arial", font_size=12, icon_path=None, icon_size=(24, 24)):
        button=QPushButton(text)
        if icon_path:
            button.setIcon(QIcon(icon_path))
            button.setIconSize(QSize(icon_size))

        button.setFixedSize(height, width)
        button.setStyleSheet(f"background-color: {bg_color}; font-family: {font_family}; font-size: {font_size};")
        button.setFont(QFont(font_family, font_size))

        return button

