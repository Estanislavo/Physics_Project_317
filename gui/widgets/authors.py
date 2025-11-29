from PyQt6 import QtWidgets, QtCore, QtGui
from config.strings import STRINGS, Lang
import os
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np


class AuthorsWidget(QtWidgets.QWidget):
    def __init__(self, parent, back_cb, get_lang_cb):
        super().__init__(parent)
        self.back_cb = back_cb
        self.get_lang_cb = get_lang_cb
        self._build_ui()

    def _load_sticker(self, fname, size=420):
        try:
            # Получаем путь к папке gui/widgets
            widget_dir = os.path.dirname(os.path.abspath(__file__))
            # Поднимаемся на два уровня выше: gui/widgets -> gui -> . (корень проекта)
            project_root = os.path.dirname(os.path.dirname(widget_dir))
            images_dir = os.path.join(project_root, "images")
            path = os.path.join(images_dir, fname)

            img = Image.open(path).convert("RGBA")
            data = np.array(img)
            mask = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
            data[mask, 3] = 0
            img = Image.fromarray(data)
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            return QtGui.QPixmap.fromImage(ImageQt(img))
        except Exception as e:
            print(e)
            return None

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Используем локализацию
        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet("font-size:32pt; font-weight:700;")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        cards = QtWidgets.QHBoxLayout()
        layout.addLayout(cards)

        left = QtWidgets.QVBoxLayout()
        left.addSpacing(80)
        left.setSpacing(6)
        left.setContentsMargins(2, 0, 2, 0)

        self.name1_label = QtWidgets.QLabel()
        self.name1_label.setStyleSheet("font-size:25pt; font-weight:600;")
        self.name1_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.name1_label)

        pix = self._load_sticker("author1.png") or self._load_sticker("author1.jpg")
        if pix:
            lbl = QtWidgets.QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(400, 500)
            lbl.setScaledContents(True)
            left.addWidget(lbl, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

        cards.addLayout(left)

        # -------- RIGHT (author 2): имя сверху, фото снизу --------
        right = QtWidgets.QVBoxLayout()
        right.addSpacing(80)
        right.setSpacing(6)
        right.setContentsMargins(2, 0, 2, 0) 

        self.name2_label = QtWidgets.QLabel()
        self.name2_label.setStyleSheet("font-size:25pt; font-weight:600;")
        self.name2_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.name2_label)

        pix2 = self._load_sticker("author2.png") or self._load_sticker("author2.jpg")
        if pix2:
            lbl2 = QtWidgets.QLabel()
            lbl2.setPixmap(pix2)
            lbl2.setFixedSize(400, 500)
            lbl2.setScaledContents(True)
            right.addWidget(lbl2, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)  # <-- ПОТОМ ФОТО

        cards.addLayout(right)

        # Сделаем колонки одинаковой ширины
        cards.setStretch(0, 1)
        cards.setStretch(1, 1)

        # -------- BOSS (по центру снизу, только текст) --------
        self.boss_label = QtWidgets.QLabel()
        self.boss_label.setStyleSheet("font-size:22pt; font-weight:600;")
        self.boss_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Прижимаем низ: сначала растяжка, затем boss, затем кнопка
        layout.addStretch(1)
        layout.addWidget(self.boss_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(12)

        self.back_btn = QtWidgets.QPushButton()
        self.back_btn.setFixedSize(200, 48)
        self.back_btn.setStyleSheet("background:#313132;color:white;font-weight:600;font-size:20pt;border-radius:6px;")
        self.back_btn.clicked.connect(self.back_cb)
        layout.addWidget(self.back_btn, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(20)

        # Устанавливаем текст при инициализации
        self.update_language(self.get_lang_cb())

    def update_language(self, lang: Lang):
        s = STRINGS[lang.value]
        self.title_label.setText(s["authors.title"])
        self.name1_label.setText(s["authors.name1"])
        self.name2_label.setText(s["authors.name2"])
        self.boss_label.setText(s["boss"])
        self.back_btn.setText(s["authors.back"])