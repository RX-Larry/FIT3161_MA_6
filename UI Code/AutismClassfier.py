import sys
import time
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow


class AutismClassifier(QMainWindow):
    def __init__(self):
        super(AutismClassifier, self).__init__()
        loadUi("AutismClassifier.ui", self)


def main():
    app = QApplication(sys.argv)
    mainScreen = AutismClassifier()
    mainScreen.show()
    mainScreen.showNormal()
    sys.exit(app.exec_())


main()
