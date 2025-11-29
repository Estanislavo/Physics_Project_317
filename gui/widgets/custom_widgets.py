from PyQt6 import QtWidgets, QtGui

class WinCheckBox(QtWidgets.QCheckBox):
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.setChecked(not self.isChecked())
        self.update()
        e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        e.accept()
