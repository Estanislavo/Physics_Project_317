import sys
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets
import logging
import matplotlib

from gui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)

    font_families = ["Microsoft YaHei", "Noto Sans CJK SC", "SimSun", "Arial"]
    for font_family in font_families:
        if QFont(font_family).exactMatch():
            font = QFont(font_family, 11)
            app.setFont(font)
            break

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').disabled = True

    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.use('Agg')

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
