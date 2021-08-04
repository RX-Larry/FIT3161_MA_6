import sys
import time
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox


class AutismClassifier(QMainWindow):
    def __init__(self):
        super(AutismClassifier, self).__init__()
        loadUi("AutismClassifier.ui", self)

        # opens about window
        self.actionAbout.triggered.connect(self.openAboutWindow)

    def openAboutWindow(self):
        aboutPage = QMessageBox()
        aboutPage.setWindowTitle("About")
        aboutPage.setIcon(QMessageBox.Information)
        aboutPage.setText("Autism Classifier")
        aboutPage.setInformativeText(
            "Project Manager: Lim Tzeyi\nTech Lead: Adrian Tham Wai Yeen\nQuality Assurance: Chong Pei Jiun\nSoftware Version: 1.0\nGitHub:"
        )

        x = aboutPage.exec_()


def main():
    app = QApplication(sys.argv)
    mainScreen = AutismClassifier()
    mainScreen.show()
    mainScreen.showNormal()
    sys.exit(app.exec_())


main()
