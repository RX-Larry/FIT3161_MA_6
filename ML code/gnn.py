# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 19:00:57 2021

@author: peiji
"""

# from spektral.datasets import TUDataset
# from spektral.transforms import GCNFilter

# dataset = TUDataset('PROTEINS')
# dataset[0]
# max_degree = dataset.map(lambda g: g.a.sum(-1).max(), reduce=max)

# dataset.apply(GCNFilter())

#%%

import numpy as np
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
    def __init__(self, n_samples, X, y, a, **kwargs):
        self.n_samples = n_samples
        self.X = X
        self.y = y
        self.a = a
        super().__init__(**kwargs)
    
    def read(self):
        # X = node features
        # a = adjacency matrix
        # y = labels

        # return a list of Graph objects
        return [Graph(x=self.X, a=self.a, y=self.y) for _ in range(self.n_samples)]

data = MyDataset(1000, transforms=NormalizeAdj())

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
