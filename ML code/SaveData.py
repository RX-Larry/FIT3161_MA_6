"""
@author: Adrian
"""

import pandas as pd
import pickle

from nilearn.datasets import fetch_abide_pcp


def load_data():
    # This will download the data that passes the quality check                                                                                                                                                                                                                       checked
    abide = fetch_abide_pcp(data_dir='data', 
                                    n_subjects=None, # Downloading all the data
                                    pipeline='cpac', 
                                    band_pass_filtering=True, 
                                    global_signal_regression=True, 
                                    derivatives=['rois_ho'],  # ROI time series files which end in .1D
                                    quality_checked=True, verbose=1)
    return abide

def preprocess_data():
    abide = load_data()
    data = abide.rois_ho
    phenotypic = abide.phenotypic
    phenotypic = pd.DataFrame(abide.phenotypic)
    
    # Decode the string columns "bytes" to utf-8 (remove b from str)
    phenotypic.columns=[col.encode('utf-8', 'replace').decode('utf-8') for col in phenotypic.columns]
    for col in phenotypic.columns:
        if type(phenotypic[col][0]) is bytes:
            phenotypic[col]= phenotypic[col].str.decode('utf-8') 
    
    # extract labels from phenotypics data
    y = abide.phenotypic['DX_GROUP'] # 2: normal, 1: autism
    return data, phenotypic, y

def save_data():
    data, phenotypic, y = preprocess_data()
    # save data, labels, and phenotypic as dictionary into a pickle file
    data_abide = {} #empty dictionary
    data_abide['timeseries'] = data
    data_abide['phenotypic'] = phenotypic
    data_abide['labels'] = y
    saveTo = 'data/abide_proc_filt_glob.pkl'
    with open(saveTo,'wb') as f:
        pickle.dump(data_abide,f)
    
    # save phenotypic as csv
    saveTo = 'data/abide_proc_filt_glob.csv'
    phenotypic.to_csv(saveTo,index=False)
    
if __name__ == "__main__":
    save_data()
    
