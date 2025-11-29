from PyQt6 import QtWidgets, QtCore, QtGui
from config.strings import STRINGS, Lang
import os
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np

class MainMenuWidget(QtWidgets.QWidget):
    def __init__(self, parent, start_cb, lang_toggle_cb, get_lang_cb, authors_cb=None, theory_cb=None):
        super().__init__(parent)
        self.start_cb = start_cb
        self.authors_cb = authors_cb  # ← сохраняем колбэк (может быть None)
        self.theory_cb = theory_cb  # ← новый колбэк для теории
        self.lang_toggle_cb = lang_toggle_cb
        self.get_lang_cb = get_lang_cb
        self._build_ui()

    def _load_img_icon(self, filename, size=(32, 22)):
        try:
            # Получаем путь к папке gui/widgets
            widget_dir = os.path.dirname(os.path.abspath(__file__))
            # Поднимаемся на два уровня выше: gui/widgets -> gui -> . (корень проекта)
            project_root = os.path.dirname(os.path.dirname(widget_dir))
            images_dir = os.path.join(project_root, "images")
            path = os.path.join(images_dir, filename)

            img = Image.open(path).convert("RGBA")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return QtGui.QIcon(QtGui.QPixmap.fromImage(ImageQt(img)))
        except Exception:
            return None

    def _load_logo_pixmap(self, filename, size=(180, 180)):
        try:
            # Получаем путь к папке gui/widgets
            widget_dir = os.path.dirname(os.path.abspath(__file__))
            # Поднимаемся на два уровня выше: gui/widgets -> gui -> . (корень проекта)
            project_root = os.path.dirname(os.path.dirname(widget_dir))
            images_dir = os.path.join(project_root, "images")
            path = os.path.join(images_dir, filename)
            img = Image.open(path).convert("RGBA")
            data = np.array(img)
            mask = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
            data[mask, 3] = 0
            img = Image.fromarray(data)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return QtGui.QPixmap.fromImage(ImageQt(img))
        except Exception:
            return None

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)

        left = QtWidgets.QVBoxLayout()
        center = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        layout.addLayout(left)
        layout.addLayout(center, stretch=1)
        layout.addLayout(right)

        # --- Левый столбец: логотип + флаг ---
        left_logo = QtWidgets.QLabel()
        pix = self._load_logo_pixmap("cmc_logo.png")
        if pix:
            left_logo.setPixmap(pix)
            left_logo.setFixedSize(180, 180)  # ФИКСИРУЕМ РАЗМЕР
            left_logo.setScaledContents(True)
        left.addWidget(
            left_logo,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        left.addStretch(1)

        # Кнопка флага (левый нижний угол)
        self.btn_flag = QtWidgets.QPushButton()
        self.btn_flag.setFixedSize(44, 32)
        self.btn_flag.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.btn_flag.setStyleSheet("border:1px solid #ccc; border-radius:6px; background:#fff;")
        self.btn_flag.clicked.connect(self.lang_toggle_cb)
        left.addWidget(
            self.btn_flag,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom
        )

        right_logo = QtWidgets.QLabel()
        pix2 = self._load_logo_pixmap("fiz_logo.png") or self._load_logo_pixmap("fiz_logo.jpg")
        if pix2:
            right_logo.setPixmap(pix2)
            right_logo.setFixedSize(180, 180)
            right_logo.setScaledContents(True)
        right.addWidget(
            right_logo,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
        )

        right.addStretch(1)

        self.lbl_title = QtWidgets.QLabel()
        self.lbl_title.setStyleSheet("font-size:37pt; font-weight:600;")
        self.lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setWordWrap(True)

        self.lbl_subtitle = QtWidgets.QLabel()
        self.lbl_subtitle.setStyleSheet("font-size:27pt; color:#444")
        self.lbl_subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_subtitle.setWordWrap(True)

        center.addStretch(1)
        center.addWidget(self.lbl_title)
        center.addWidget(self.lbl_subtitle)

        self.btn_start = QtWidgets.QPushButton()
        self.btn_start.setMinimumHeight(60)
        self.btn_start.setMinimumWidth(250)
        self.btn_start.setStyleSheet(
            "background:#3271a8; color:white; font-weight:600; font-size:20pt; border-radius:6px; margin-top:10px; padding: 6px 12px;"
        )
        self.btn_start.clicked.connect(self.start_cb)
        center.addWidget(self.btn_start, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        if hasattr(self, "theory_cb") and self.theory_cb:
            self.btn_theory = QtWidgets.QPushButton()
            self.btn_theory.setMinimumHeight(60)
            self.btn_theory.setMinimumWidth(250)
            self.btn_theory.setStyleSheet(
                "background:#5a8d5a; color:white; font-weight:600; font-size:20pt; border-radius:6px; margin-top:10px; padding: 6px 12px;"
            )
            self.btn_theory.clicked.connect(self.theory_cb)
            center.addWidget(self.btn_theory, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        if hasattr(self, "authors_cb") and self.authors_cb:
            self.btn_authors = QtWidgets.QPushButton()
            self.btn_authors.setMinimumHeight(60)
            self.btn_authors.setMinimumWidth(250)
            self.btn_authors.setStyleSheet(
                "background:#a6b2bd; color:white; font-weight:600; font-size:20pt; border-radius:6px; margin-top:10px; padding: 6px 12px;"
            )
            self.btn_authors.clicked.connect(self.authors_cb)
            center.addWidget(self.btn_authors, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_exit = QtWidgets.QPushButton()
        self.btn_exit.setMinimumHeight(60)
        self.btn_exit.setMinimumWidth(250)
        self.btn_exit.setStyleSheet(
            "background:#f78765; color:white; font-weight:600; font-size:20pt; border-radius:6px; margin-top:10px; padding: 6px 12px;"
        )
        self.btn_exit.clicked.connect(QtWidgets.QApplication.instance().quit)
        center.addWidget(self.btn_exit, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        center.addStretch(2)

        self.update_language(self.get_lang_cb())

    def update_language(self, lang: Lang):
        s = STRINGS[lang.value]
        self.lbl_title.setText(s["menu.title"])
        self.lbl_subtitle.setText(s["menu.subtitle"])
        self.btn_start.setText(s["menu.start"])
        self.btn_exit.setText(s["menu.exit"])

        if lang == Lang.RU:
            icon_file = "rus.png"
        elif lang == Lang.EN:
            icon_file = "eng.png"
        elif lang == Lang.CN:
            icon_file = "china.png"
        else:
            icon_file = "rus.png"

        icon = self._load_img_icon(icon_file)
        if icon:
            self.btn_flag.setIcon(icon)
            self.btn_flag.setIconSize(QtCore.QSize(28, 20))
        if hasattr(self, 'btn_authors') and self.btn_authors is not None:
            self.btn_authors.setText(s.get('menu.authors', 'Об авторах'))
        if hasattr(self, 'btn_theory') and self.btn_theory is not None:
            self.btn_theory.setText(s.get('menu.theory', 'Теория'))
