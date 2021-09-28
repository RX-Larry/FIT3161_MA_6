# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:46:34 2021

@author: peijiun
"""

import os
import sklearn
from SaveData import  load_data, preprocess_data, save_data

def test_load_data():
    total_file = 871
    filepath = os.getcwd() + "\data\ABIDE_pcp\cpac\\filt_global"
    if not os.path.exists(filepath):    
        abide = load_data()
        assert isinstance(abide, sklearn.utils.Bunch) == True
    count = 0
    for fname in os.listdir(filepath):
        if fname.endswith('.1D'):
            count += 1
    assert count == total_file # after filtering out lower quality data, left with 871 data

def test_preprocess_data():
    data, phenotypic, y = preprocess_data()
    
    assert len(data) == 871
    assert phenotypic.shape == (871, 106)
    assert y.shape == (871,)

def test_save_data():
    save_data() # after executing this function, the data file should be saved
    filepath_pkl = os.getcwd() + "\data\\abide_proc_filt_glob.pkl"
    filepath_csv = os.getcwd() + "\data\\abide_proc_filt_glob.csv"
    
    assert os.path.exists(filepath_pkl) == True
    assert os.path.exists(filepath_csv) == True

if __name__ == "__main__":
    test_load_data()
    test_preprocess_data()
    test_save_data()