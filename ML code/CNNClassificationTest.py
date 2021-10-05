# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:00:44 2021

@author: peijiun
"""

from CNNClassification import test_model

def test_test_model():
    
    acc, Pre, Rec, F1, ROC = test_model()
    
    assert acc >= 0.8
    assert Pre >= 0.8
    assert Rec >= 0.8
    assert F1 >= 0.8
    assert ROC >= 0.8
    
    
    
if __name__ == "__main__":
    test_test_model()