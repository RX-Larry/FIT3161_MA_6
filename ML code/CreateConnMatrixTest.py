# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:45:27 2021

@author: peijiun
"""

import os

from CreateConnMatrix import load_timeseries, create_conn_matrix

def test_load_timeseries():
    ts_data, labels = load_timeseries()
    
    assert len(ts_data) == 871
    assert all(elem == 0 or elem == 1 for elem in labels) == True # make sure the elem only contains 0 and 1

def test_create_conn_matrix():
    filepath_conn_matrix_pkl = os.getcwd() + "\data\\connectivity_matrices.pkl"
    conn_matrix = create_conn_matrix()
    
    assert conn_matrix.shape == (871, 111, 111)
    assert os.path.exists(filepath_conn_matrix_pkl) == True

if __name__ == "__main__":
    test_load_timeseries()
    test_create_conn_matrix()