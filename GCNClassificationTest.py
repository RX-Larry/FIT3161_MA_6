# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:05:17 2021

@author: peijiun
"""
import pytest
import numpy as np
import pandas as pd
import stellargraph

from step4_GCN_Classification import threshold_proportional, conv2list, build_graphs

def test_threshold_proportional():
    W = np.array([[-0.54190427, -0.27866048,  0.455306,   -0.77466439,  0.2155413 ],
         [ 0.63149892,  0.96253877, -0.87251032,  0.5999195,  -0.80610289],
         [-0.1982645,  0.32431534, 0.93117182, -0.03819988, -0.47177543],
         [ 0.17483424, -0.88284286,  0.19139394, -0.11495341,  0.06681537],
         [ 0.18449563, -0.18105407,  0.40700154, -0.92213003, -0.79312868]])
    
    W_thr = np.array([[0., 0., 0.455306  , 0., 0.],
                     [0.63149892, 0., 0., 0.5999195 , 0.],
                     [0., 0.32431534, 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0.40700154, 0., 0.]])
    
    threshold = [0.25, -0.1, 1.5]
    ERROR_MSG = "Threshold value should be between 0 and 1"
    
    for i in threshold:
        if i < 0 or i > 1:
            with pytest.raises(ValueError) as e:
                threshold_proportional(W, i)
            assert ERROR_MSG in str(e)
        else:
            assert (np.allclose(threshold_proportional(W, i), W_thr)) == True # check if original matrix is same as matrix after thresholding
            
def test_conv2list():
    adj_thr = np.array([[0.22904787, 0.36066976],
               [0.727653,   0.1126678 ]])
    adj_thr_err = np.array([])
    data = {
        'source': [0.0, 0.0, 1.0, 1.0],
        'target': [0.0, 1.0, 0.0, 1.0],
        'weight': [0.22904787, 0.36066976, 0.727653, 0.1126678]
        }
    df = pd.DataFrame(data)
    df2 = conv2list(adj_thr)
    
    ERROR_MSG = "Invalid adjacency matrix"
    
    assert (df.equals(df2)) == True
    with pytest.raises(ValueError) as e:
        conv2list(adj_thr_err)
    assert ERROR_MSG in str(e)
    
def test_build_graphs():
    conn_mat = np.array([[[0.22904787, 0.36066976, 0.727653,   0.1126678,  0.60777065],
                          [0.81574946, 0.98126938, 0.06374484, 0.79995975, 0.09694856],
                          [0.40086775, 0.66215767, 0.96558591, 0.48090006, 0.26411229],
                          [0.58741712, 0.05857857, 0.59569697, 0.4425233 , 0.53340769],
                          [0.59224781, 0.40947296, 0.70350077, 0.03893498, 0.10343566]],
                        
                         [[0.28225711, 0.1531149,  0.30005337, 0.31885786, 0.69653026],
                          [0.67662691, 0.94246392, 0.40874647, 0.58077894, 0.70695474],
                          [0.10881165, 0.39721578, 0.34467615, 0.62997804, 0.36956759],
                          [0.15654754, 0.07882936, 0.85880122, 0.91120746, 0.83249014],
                          [0.62265741, 0.28735161, 0.4012328,  0.6185372 , 0.78173855]]])
    
    for i in build_graphs(conn_mat, conn_mat):
        assert isinstance(i, stellargraph.core.graph.StellarGraph) == True
    
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    test_threshold_proportional()
    test_conv2list()
    test_build_graphs()


    

        
