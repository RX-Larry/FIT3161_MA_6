import pytest
from PyQt5 import QtCore
import AutismClassifier

@pytest.fixture
def app(qtbot):
    test_app = AutismClassifier.AutismClassifier()
    qtbot.addWidget(test_app)

    return test_app


def test_no_file_checking(app, qtbot):
    qtbot.mouseClick(app.runProgramButton, QtCore.Qt.LeftButton)
    assert app.fileErrorLabel.text() == "Please insert a nii.gz format compressed fMRI data file"

def test_wrong_file_format_checking(app, qtbot):
    test_file = ('test.png','test')
    app.filename=test_file
    qtbot.mouseClick(app.runProgramButton, QtCore.Qt.LeftButton)
    assert app.fileErrorLabel.text() == "The file format must be nii.gz"