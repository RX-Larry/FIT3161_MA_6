# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 19:00:57 2021

@author: peijiun
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

import scipy.sparse as sp
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import BatchLoader, Dataset, DisjointLoader, Graph
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers.pooling import TopKPool

from sklearn.model_selection import train_test_split

#%%

# Load data from the pickle file
datapath = 'data/connectivity_matrices.pkl'
with open(datapath,'rb') as f:
    conn_data = pickle.load(f)

tangent_matrices = conn_data['FC']
labels = conn_data['labels']

threshold = 0.15
lower=0
upper=1
for ij in np.ndindex(tangent_matrices.shape):
    tangent_matrices[ij] = np.where(tangent_matrices[ij]>threshold, upper, lower)

# tangent matrix plot
# plotting.plot_matrix(tangent_matrices[0], figure=(50, 50), labels=range(111),
#                      vmax=0.8, vmin=-0.8, reorder=True)

n_samples = labels.shape[0]
y=labels
X=np.array([i for i in range(111)])

#%%

learning_rate = 1e-2  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 10  # Patience for early stopping
batch_size = 32  # Batch size

#%%

class MyDataset(Dataset):
    def __init__(self, n_samples,X, y, a, **kwargs):
        self.n_samples = n_samples
        self.nodeFeatures = X
        self.labels = y
        self.adjMatrix = a
        super().__init__(**kwargs)
    
    def read(self):
        # X = node features
        # a = adjacency matrix
        # y = labels
        
        # return a list of Graph objects
        graphList = []
        a = self.adjMatrix.astype(int)
        
        X = self.nodeFeatures
        y = self.labels
        # for _ in range(self.n_samples):
        for node in a:
            graphList.append(Graph(x=X, a=node, y=y))
        return graphList
    
# Train/valid/test split
def train_valid_test_split(data, target, train_size, test_size):
    valid_size = 1 - (train_size + test_size)
    X1, X_test, y1, y_test = train_test_split(data, target, test_size = test_size, random_state= 33)
    X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size = float(valid_size)/(valid_size+ train_size))
    return X_train, X_valid, X_test, y_train, y_valid, y_test

X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(tangent_matrices, y, train_size=0.8, test_size=0.1)

data_tr = MyDataset(X_train.shape, X, y_train, a=X_train, transforms=NormalizeAdj())
data_va = MyDataset(X_valid.shape, X, y_valid, a=X_valid, transforms=NormalizeAdj())
data_te = MyDataset(X_test.shape, X, y_test, a=X_test, transforms=NormalizeAdj())

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)
#%%

class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, 'softmax')
        
        # self.conv2 = GCNConv(n_hidden, activation='relu')
        # self.pool2 = TopKPool(ratio=0.5)
        # self.dropout2 = Dropout(0.5)
        
        # self.conv3 = GCNConv(n_hidden, activation='relu')
        self.pool = GlobalSumPool(ratio=0.5)
        # self.dropout3 = Dropout(0.5)
        
        self.dense = Dense(n_labels, activation='softmax')

    def call(self, inputs):
        # x, a, i = inputs
        # x = self.conv1([x, a])
        # x1, a1, i1 = self.pool1([x, a, i])
        # x1 = self.conv2([x1, a1])
        # x2, a2, i2 = self.pool2([x1, a1, i1])
        # x2 = self.conv3([x2, a2])
        # out = self.poo3([x2, i2])
        # out = self.dense(out)
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)
        return out
    
model = MyFirstGNN(32, 2)
optimizer = Adam(learning_rate=learning_rate)
loss_fn = BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=epochs)

model.summary()

loss = model.evaluate(loader_va.load(), steps=loader_va.steps_per_epoch)
predict = model.predict(loader_te.load(), steps=loader_te.steps_per_epoch)
print('Test loss: {}'.format(loss))
