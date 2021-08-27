"""
@author: Adrian
"""

import pickle
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt

# Load data from the pickle file
datapath = 'data/abide_proc_filt_glob.pkl'
with open(datapath,'rb') as f:
    data = pickle.load(f)

ts_data = data['timeseries'] 
labels =  data['labels']
#Converting normal to 0 and autism to 1.
labels[labels>1]=0

# =============================================================================
# # pearson correlation
# connectivity_estimator = ConnectivityMeasure(kind='correlation')
# 
# # Fit the covariance estimator to the given time series for each subject. Then apply transform to covariance matrices for the chosen kind.
# connectivity_matrices = connectivity_estimator.fit_transform(ts_data)
# 
# from nilearn import plotting   
# plotting.plot_matrix(connectivity_matrices[0], labels=range(1,112), colorbar=True,
#                  vmax=0.8, vmin=-0.8)
# =============================================================================

# Tangent
tangent_estimator = ConnectivityMeasure(kind='correlation')
tangent_matrices = tangent_estimator.fit_transform(ts_data)

#Plot the correlations for the first 40 subjects
for i in range(40):
    plt.figure(figsize=(8,6))
    plt.imshow(tangent_matrices[i], vmax=.80, vmin=-.80, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Connectivity matrix of subject {} with label {}'.format(i, labels[i]))

# save the connectivity matrices and labels as a dictionary into a pickle file   
conn_data = {}
conn_data['FC'] = tangent_matrices
conn_data['labels'] = labels

saveTo = 'data/connectivity_matrices.pkl'
with open(saveTo,'wb') as f:
    pickle.dump(conn_data,f)
    