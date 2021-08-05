import sys
import time
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox


class AutismClassifier(QMainWindow):
    def __init__(self):
        super(AutismClassifier, self).__init__()
        loadUi("AutismClassifier.ui", self)

        # opens file explorer when browse file button pressed
        self.browseFilesButton.clicked.connect(self.browseFiles)

        # opens about window
        self.actionAbout.triggered.connect(self.openAboutWindow)

        # for progress bar sample look when "Run Program" button is pressed
        n = 100
        self.runProgramButton.clicked.connect(
            lambda status, n_size=n: self.progressBarLoading(n_size)
        )

    def openAboutWindow(self):
        aboutPage = QMessageBox()
        aboutPage.setWindowTitle("About")
        aboutPage.setIcon(QMessageBox.Information)
        aboutPage.setText("Autism Classifier")
        aboutPage.setInformativeText(
            "Project Manager: Lim Tzeyi\nTech Lead: Adrian Tham Wai Yeen\nQuality Assurance: Chong Pei Jiun\nSoftware Version: 1.0\nGitHub:"
        )

        x = aboutPage.exec_()

    def browseFiles(self):
        fileName = QFileDialog.getOpenFileName(self, "Open file", "C:/Users")
        self.filePathDisplay.setText(fileName[0])

    def progressBarLoading(self, n):
        for i in range(n):
            time.sleep(0.01)
            self.progressBar.setValue(i + 1)


def main():
    app = QApplication(sys.argv)
    mainScreen = AutismClassifier()
    mainScreen.show()
    mainScreen.showNormal()
    sys.exit(app.exec_())


main()
