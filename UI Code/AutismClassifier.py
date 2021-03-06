import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import nibabel as nib
from nilearn import decomposition, plotting, input_data
from nilearn.connectome import ConnectivityMeasure
import os
from stellargraph import StellarGraph
import pandas as pd
from stellargraph.mapper import PaddedGraphGenerator
import tensorflow as tf
import numpy as np

img = None


def threshold_proportional(W, p, copy=True):
    """
    Convert values less than the threshold value to 0

    Parameters
    ----------
    W : 2D array, connevtivity matrix to be thresholded.
    p : float value between 0 and 1, Cell Value less than threshold value will be set to 0.
    copy : boolean, optional, The default is True.

    Raises
    ------
    ValueError, If the threshold is not within 0 and 1.

    Returns
    -------
    W : Thresholded 2D array, A matrix that does not contains negative values.

    """
    assert p < 1 or p > 0
    if copy:
        W = W.copy()
    n = len(W)  # number of nodes
    np.fill_diagonal(W, 0)  # clear diagonal
    if np.all(W == W.T):  # if symmetric matrix
        W[np.tril_indices(n)] = 0  # ensure symmetry is preserved
        ud = 2  # halve number of removed links
    else:
        ud = 1
    ind = np.where(W)  # find all links
    I = np.argsort(W[ind])[::-1]  # sort indices by magnitude
    # number of links to be preserved
    en = round((n * n - n) * p / ud)
    W[(ind[0][I][en:], ind[1][I][en:])] = 0  # apply threshold
    if ud == 2:  # if symmetric matrix
        W[:, :] = W + W.T  # reconstruct symmetry

    W[W > 0.9999] = 1  # make sure the highest correlation coeff is 1
    return W


def conv2list(adj_m):
    """
    converts adjacency matrix to adj list to load into stellargraph

    Parameters
    ----------
    adj_m : 2D array to be converted to adjacency list.

    Raises
    ------
    ValueError
        if connectivity matrix has length 0.

    Returns
    -------
    d : DataFrame.

    """

    # find non-zero elements in adj_mat
    indices = np.argwhere(adj_m)
    src, dsts = indices[:, 0].reshape(-1, 1), indices[:, 1].reshape(-1, 1)
    v = adj_m[src, dsts].reshape(-1, 1)
    final = np.concatenate((src, dsts, v), axis=1)
    d = pd.DataFrame(final)
    d.columns = ["source", "target", "weight"]

    return d


def build_graphs(node_feat, adj_data):
    """
    Convert adjacency list to graphs

    Parameters
    ----------
    node_feat : 2D array
        where each row represent a node features for the brain region.
    adj_data : 2D array
        adjacency matrix.

    Raises
    ------
    ValueError
        if all the nodes are not connected.

    Returns
    -------
    graphs : Stellargraph Object
        to be used as an input later to the GNN model.

    """
    graphs = []
    min_T = np.min([item.shape[1] for item in node_feat])  # assuming last dim is Time
    for A, X in zip(adj_data, node_feat):
        A_thr = threshold_proportional(A, 0.25)  # adjacency matrix
        np.fill_diagonal(A_thr, 1)  # add selve-connectins to avoid zero in-degree nodes
        assert np.sum(A_thr, axis=0).all()
        A_df = conv2list(A_thr)
        timeseries = X[:min_T]  # node features (ROI,Time)
        X_df = pd.DataFrame(timeseries)
        G = StellarGraph(X_df, A_df)
        graphs.append(G)

    return graphs

class Worker(QObject):
    '''
    A worker class that will run the model classification code.

    Attributes
    ----------
    finished: pyqtSignal
        An pyqtSignal object that sends a finished signal to the main thread.
    progress: pyqtSignal
        An pyqtSignal object that sends the progress status to the main thread.
    prediction: pyqtSignal
        An pyqtSignal object that sends the progress messages and classification results to the main thread.
    showPlot: pyqtSignal
        An pyqtSignal object that sends a  signal to the main thread to display the brain connectome.
    '''
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    prediction = pyqtSignal(str)
    showPlot = pyqtSignal()

    def run(self):
        '''
        A method that contains the model classification code which will be run by worker thread.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        '''
        global img

        self.prediction.emit("Applying CanICA on the fMRI data...")

        canica = decomposition.CanICA(
            n_components=111, verbose=10, mask_strategy="template", random_state=0
        )

        self.progress.emit(1)

        canica.fit(img)

        self.progress.emit(90)

        self.prediction.emit(
            "Computing the corresponding linear component combination in whole-brain voxel space..."
        )

        # Using a masker to project into the 3D space.
        components = canica.masker_.inverse_transform(canica.components_)

        self.progress.emit(91)

        self.prediction.emit("Extracting the brain regions time series...")

        # Using a filter to extract the regions time series.
        masker = input_data.NiftiMapsMasker(
            components,
            smoothing_fwhm=6,
            standardize=False,
            detrend=True,
            t_r=2.5,
            low_pass=0.1,
            high_pass=0.01,
        )

        # Extracting the time series of the user input fMRI data.
        time_series = [masker.fit_transform(img)]

        self.progress.emit(94)

        self.prediction.emit("Building the connectivity matrix from the time series...")

        # Building the correlation matrix from the time series.
        correlation_measure = ConnectivityMeasure(kind="correlation")
        correlation_matrix = correlation_measure.fit_transform(time_series)
        # Plotting and saving the connectivity matrix.
        connectivity_plot = plotting.plot_matrix(
            correlation_matrix[0],
            labels=range(1, 112),
            colorbar=True,
            vmax=0.8,
            vmin=-0.8,
        )
        connectivity_plot.figure.savefig("matrix.png")

        # Getting the center coordinates from the component decomposition to use as atlas labels.
        coords = plotting.find_probabilistic_atlas_cut_coords(components)
        # Plotting, saving and displaying the connectome with 80% edge strength in the connectivity.
        connectome_plot = plotting.plot_connectome(
            correlation_matrix[0],
            coords,
            edge_threshold="50%",
            title="Correlation between 111 brain regions",
        )
        connectome_plot.savefig("connectome.png")
        self.showPlot.emit()

        self.progress.emit(97)

        self.prediction.emit("Predicting the result...")

        # Convert the connectivity matrix into a graph
        graphs = build_graphs([correlation_matrix[0]], [correlation_matrix[0]])
        generator = PaddedGraphGenerator(graphs=graphs)
        test_gen = generator.flow([0])
        # Loading the model
        saved_model = tf.keras.models.load_model("model/gcn_model")
        # Predicting the label of the user input fMRI
        result = saved_model.predict(test_gen, verbose=0).squeeze()
        # Displaying the result of the classification onto the UI.
        result_text = (
            "Result: The probability that Autism Spectrum Disorder Category was detected in the subject is "
            + "{:.0%}".format(result)
            + "."
        )
        if result >= 0.5:
            self.prediction.emit(
                result_text
                + " \nTherefore, it can be concluded that the subject has Autism Spectrum Disorder."
            )
        else:
            self.prediction.emit(
                result_text
                + " \nTherefore, it can be concluded that the subject does not have Autism Spectrum Disorder."
            )

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

        # opens user guide window
        self.actionUser_Guide.triggered.connect(self.openUserGuideWindow)

        # connecting a method that implements the progress bar's logic to the progress bar
        n = 100
        self.runProgramButton.clicked.connect(
            lambda status, n_size=n: self.progressBarLoading(n_size)
        )
        self.filename = None

        # Adding visualisation options to the combo box.
        vis_list = [
            "",
            "Connectivity Matrix",
            "Connectome on Top of The Brain Class Schematics",
        ]
        self.visComboBox.addItems(vis_list)
        self.visComboBox.setDisabled(True)
        self.visComboBox.activated.connect(self.selectVisToDisplay)

    def openAboutWindow(self):
        aboutPage = QMessageBox()
        aboutPage.setWindowTitle("About")
        aboutPage.setIcon(QMessageBox.Information)
        aboutPage.setText('<p style="font-size:11pt;">Autism Classifier</p>')
        aboutPage.setInformativeText(
            "Project Manager: Lim Tzeyi\nTech Lead: Adrian Tham Wai Yeen\nQuality Assurance: Chong Pei Jiun\nSoftware Date: \nGitHub: https://github.com/RX-Larry/FIT3161_MA_6"
        )
        aboutPage.exec_()

    def openUserGuideWindow(self):
        guidePage = QMessageBox()
        guidePage.setWindowTitle("User Guide")
        guidePage.setIcon(QMessageBox.Question)
        guidePage.setText('<p style="font-size:13pt;">User Guide</p>')
        guidePage.setInformativeText(
            '<p style="font-size:12pt;">Steps to start the classification process:</p> <p style="font-size:11pt;">1. Click on the "Browse Files" button.</p> <p style="font-size:11pt;">2. Navigate to the location that contains the fMRI data file in .nii.gz format for classification using the popup File Explorer.</p> <p style="font-size:11pt;">3. Open the file and the path to the selected file should be displayed in the file path text box.</p> <p style="font-size:11pt;">4. Once the fMRI data file is opened in the app, click on the "Run Program" button and the classification process will be started. It should take at least 9 minutes for the classification process to complete.</p> <p style="font-size:12pt;">How to select a visualisation to view?</p> <p style="font-size:11pt;">Once the classification process is completed, you can click on the Visualisation dropdown menu to select the desired visualisation to view.</p> <p style="font-size:11pt;">Note: Please refer to the user guide document for a more detailed user guide.</p>'
        )
        guidePage.exec_()

    # A method that implements the functionality of openning a file for the "Browse Files" button.
    def browseFiles(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open file", "C:/Users")
        self.filePathDisplay.setText(self.filename[0])

    # A method that implements the login for the "Run Program" button.
    def progressBarLoading(self, n):
        self.resetComponents()
        global img

        # If the user did not insert any file to the app and he pressed 'Run program', then display error message
        if self.filename == None or self.filename[0] == "":
            self.fileErrorLabel.setText(
                "Please insert a nii.gz format compressed fMRI data file"
            )
        else:
            # If the format of the file insert by the user is not nii.gz and he pressed 'Run program', then display error message
            if self.filename[0].endswith(".nii.gz") == False:
                self.fileErrorLabel.setText("The file format must be nii.gz")
            # If the user inserts a raw compressed fMRI data file in .nii.gz format, then open it and do the prediction
            else:
                self.fileErrorLabel.clear()
                img = nib.load(self.filename[0])

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
                self.worker.showPlot.connect(self.showPlot)

                # Start the thread
                self.thread.start()

                # Disable the buttons
                self.runProgramButton.setDisabled(True)
                self.browseFilesButton.setDisabled(True)
                self.visComboBox.setDisabled(True)
                self.visComboBox.setCurrentText("")

                # Enabling back the buttons and combo box after the model finished predicting
                self.thread.finished.connect(
                    lambda: self.runProgramButton.setDisabled(False)
                )
                self.thread.finished.connect(
                    lambda: self.browseFilesButton.setDisabled(False)
                )
                self.thread.finished.connect(
                    lambda: self.visComboBox.setDisabled(False)
                )

    # A method that updates the progress of the progress bar.
    def updateProgress(self, i):
        self.progressBar.setValue(i)

    # A method that resets the progress bar and result text state.
    def resetComponents(self):
        self.fileErrorLabel.clear()
        self.diagram.clear()
        self.progressBar.setValue(0)
        self.resultLabel.clear()
        self.visComboBox.setDisabled(True)
        self.visComboBox.setCurrentText("")
        for vis_file in ["matrix.png", "connectome.png"]:
            if os.path.exists(vis_file):
                os.remove(vis_file)

    # A method that updates the result text with the outcome of the prediction.
    def updatePredictionResult(self, result):
        self.resultLabel.setText(result)

    # A method that displays the connectivity matrix plot on the app.
    def showPlot(self):
        pixmap = QPixmap("connectome.png")
        self.diagram.setPixmap(pixmap)
        self.diagram.resize(700, 500)

    # A method that display the visualisation selected by the user from the combo box.
    def selectVisToDisplay(self):
        current_vis = self.visComboBox.currentText()
        # If the selected visualisation is the connectivity matrix then display it.
        if current_vis == "Connectivity Matrix":
            pixmap = QPixmap("matrix.png")
        # If the selected visualisation is the connectome then display it.
        elif current_vis == "Connectome on Top of The Brain Class Schematics":
            pixmap = QPixmap("connectome.png")

        # Only display the visualisation if it is either the connectivity matrix or the connectome.
        if current_vis in [
            "Connectivity Matrix",
            "Connectome on Top of The Brain Class Schematics",
        ]:
            self.diagram.setPixmap(pixmap)
            self.diagram.resize(700, 500)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainScreen = AutismClassifier()
    mainScreen.show()
    mainScreen.showNormal()
    sys.exit(app.exec_())
