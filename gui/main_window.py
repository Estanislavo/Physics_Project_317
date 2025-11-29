import os
import sys
from PyQt6 import QtWidgets, QtCore, QtGui
from config.strings import STRINGS, Lang
from gui.widgets.main_menu import MainMenuWidget
from gui.widgets.simulation import SimulationWidget
from gui.widgets.authors import AuthorsWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = Lang.RU

        self._build_ui()
        self.apply_language()
        self.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowFullScreen)

    def _build_ui(self):
        central = QtWidgets.QStackedWidget()
        self.setCentralWidget(central)

        # Pages
        self.menu = MainMenuWidget(
            self,
            start_cb=self.show_sim,
            lang_toggle_cb=self.toggle_language,
            get_lang_cb=lambda: self.lang,
            authors_cb=self.show_authors,
            theory_cb=self.open_theory_pdf  # ← добавляем колбэк для теории
        )
        self.sim = SimulationWidget(self, back_cb=self.show_menu, get_lang_cb=lambda: self.lang)
        self.authors = AuthorsWidget(self, back_cb=self.show_menu, get_lang_cb=lambda: self.lang)

        central.addWidget(self.menu)
        central.addWidget(self.sim)
        central.addWidget(self.authors)
        self.stack = central

        self.setWindowTitle(STRINGS[self.lang.value]["app.title"])
        self.show_menu()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu)

    def show_sim(self):
        self.stack.setCurrentWidget(self.sim)

    def show_authors(self):
        self.stack.setCurrentWidget(self.authors)

    def open_theory_pdf(self):
        """Открывает PDF-файл с теорией"""
        try:
            pdf_path = os.path.join(os.getcwd(), "theory.pdf")
            if os.path.exists(pdf_path):
                if sys.platform == "win32":
                    os.startfile(pdf_path)
                elif sys.platform == "darwin":  # macOS
                    os.system(f'open "{pdf_path}"')
                else:  # linux
                    os.system(f'xdg-open "{pdf_path}"')
            else:
                # Если файл не найден, показываем сообщение
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                msg.setWindowTitle("Файл не найден")
                msg.setText("Файл theory.pdf не найден в папке с программой.")
                msg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл теории: {str(e)}")

    def set_language(self, lang: Lang):
        self.lang = lang
        self.apply_language()

    def toggle_language(self):
        # Циклическое переключение: RU → EN → CN → RU → ...
        if self.lang == Lang.RU:
            self.lang = Lang.EN
        elif self.lang == Lang.EN:
            self.lang = Lang.CN
        else:  # CN
            self.lang = Lang.RU
        self.apply_language()

    def apply_language(self):
        s = STRINGS[self.lang.value]
        self.setWindowTitle(s["app.title"])
        if hasattr(self.menu, "update_language"):
            self.menu.update_language(self.lang)
        if hasattr(self.sim, "update_language"):
            self.sim.update_language(self.lang)
        if hasattr(self.authors, "update_language"):
            self.authors.update_language(self.lang)