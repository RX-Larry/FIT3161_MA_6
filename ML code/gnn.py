# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 19:00:57 2021

@author: peijiun
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

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

import scipy.sparse as sp
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import BatchLoader, Dataset, DisjointLoader, Graph
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers.pooling import TopKPool

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

data = MyDataset(871, X, y, a=tangent_matrices, transforms=NormalizeAdj())


# Train/valid/test split
idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

n_labels = data.n_samples
#%%

class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.conv1 = GCNConv(n_hidden, activation='relu')
        self.pool1 = TopKPool(ratio=0.5)
        self.dropout1 = Dropout(0.5)
        
        self.conv2 = GCNConv(n_hidden, activation='relu')
        self.pool2 = TopKPool(ratio=0.5)
        self.dropout2 = Dropout(0.5)
        
        self.conv3 = GCNConv(n_hidden, activation='relu')
        self.pool3 = GlobalSumPool(ratio=0.5)
        self.dropout3 = Dropout(0.5)
        
        self.dense = Dense(n_labels, activation='softmax')

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x1, a1, i1 = self.pool1([x, a, i])
        x1 = self.conv2([x1, a1])
        x2, a2, i2 = self.pool2([x1, a1, i1])
        x2 = self.conv3([x2, a2])
        out = self.poo3([x2, i2])
        out = self.dense(out)
        return out
    
model = MyFirstGNN(32, n_labels)
optimizer = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()

# @tf.function(input_signature=loader_tr.tf_signature())  # Specify signature here
# def train_step(inputs, target):
#     # keep track of our gradient
#     with tf.GradientTape() as tape: # automatic differentiation
#         predictions = model(inputs, training=True)
#         loss = loss_fn(target, predictions)
#     # calculate the gradients using our tape and then update the model weights
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# for batch in loader_tr:
#     train_step(*batch)

model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=100)

model.summary()

loss = model.evaluate(loader_va.load(), steps=loader_va.steps_per_epoch)
predict = model.predict(loader_te.load(), steps=loader_te.steps_per_epoch)
print('Test loss: {}'.format(loss))
