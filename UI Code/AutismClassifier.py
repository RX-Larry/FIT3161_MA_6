import sys
import time
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import nibabel as nib
from nilearn import decomposition,plotting,image,input_data
from nilearn.connectome import ConnectivityMeasure
from tensorflow.keras.models import load_model

img=None

# A worker class that will contain the model prediction code.
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    prediction = pyqtSignal(str)

    # A run method that contains all the model prediction related code.
    def run(self):
        global img
        canica = decomposition.CanICA(n_components=111,
                verbose=10,
                mask_strategy='template',
                random_state=0)

        self.progress.emit(1)
        
        canica.fit(img)

        self.progress.emit(90)

        # Using a masker to project into the 3D space
        components = canica.masker_.inverse_transform(canica.components_)

        self.progress.emit(91)

        # Using a filter to extract the regions time series 
        masker = input_data.NiftiMapsMasker(components, smoothing_fwhm=6,
                                            standardize=False, detrend=True,
                                            t_r=2.5, low_pass=0.1,
                                            high_pass=0.01)

        # Extracting the time series of the user input fMRI data
        time_series = [masker.fit_transform(img)]

        self.progress.emit(94)

        #Building the correlation matrix from the time series
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform(time_series)

        self.progress.emit(97)

        # This model part here is just for testing the predicting functionality, will be replace with GNN model after the GNN model is done.
        correlation_matrix  = correlation_matrix[...,None]
        model = load_model('C:/Users/ishoo/Desktop/UNI/FYP 2/Code stuff/best_model.hdf5')
        result = model.predict(x=correlation_matrix,verbose=0).squeeze()
        if result >=0.5:
            self.prediction.emit("Yes")
        else:
            self.prediction.emit("No")

        self.progress.emit(100)

        self.finished.emit()

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
        self.filename=None

    def openAboutWindow(self):
        aboutPage = QMessageBox()
        aboutPage.setWindowTitle("About")
        aboutPage.setIcon(QMessageBox.Information)
        aboutPage.setText("Autism Classifier")
        aboutPage.setInformativeText(
            "Project Manager: Lim Tzeyi\nTech Lead: Adrian Tham Wai Yeen\nQuality Assurance: Chong Pei Jiun\nSoftware Version: 1.0\nGitHub:"
        )

        x = aboutPage.exec_()
    
    # A method that implements the functionality of openning a file for the "Browse Files" button.
    def browseFiles(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open file", "C:/Users")
        self.filePathDisplay.setText(self.filename[0])

    # A method that implements the login for the "Run Program" button.
    def progressBarLoading(self, n):
        self.resetComponents()
        global img

        # Opens the raw fMRI data file in .nii format
        if (self.filename!=None):
            img = nib.load(self.filename[0])

        # https://realpython.com/python-pyqt-qthread/#communicating-with-worker-qthreads
        # Creating a QThread object for the background thread
        self.thread = QThread()
        
        # Creating a worker object that will run the model prediction code
        self.worker = Worker()
        
        # Move worker to the thread
        self.worker.moveToThread(self.thread)
        
        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.updateProgress)
        self.worker.prediction.connect(self.updatePredictionResult)
        
        # Start the thread
        self.thread.start()

        # Disable the buttons
        self.runProgramButton.setDisabled(True)
        self.browseFilesButton.setDisabled(True)

        # Enabling back the buttons after the model finished predicting
        self.thread.finished.connect(
            lambda: self.runProgramButton.setDisabled(False)
        )
        self.thread.finished.connect(
            lambda: self.browseFilesButton.setDisabled(False)
        )

    # A method that updates the progress of the progress bar.
    def updateProgress(self,i):
        self.progressBar.setValue(i)

    # A method that resets the progress bar and result text state.
    def resetComponents(self):
        self.progressBar.setValue(0)
        self.resultLabel.setText("Result: ")

    # A method that updates the result text with the outcome of the prediction.
    def updatePredictionResult(self,result):
        self.resultLabel.setText("Result: "+result)

def main():
    app = QApplication(sys.argv)
    mainScreen = AutismClassifier()
    mainScreen.show()
    mainScreen.showNormal()
    sys.exit(app.exec_())


main()
