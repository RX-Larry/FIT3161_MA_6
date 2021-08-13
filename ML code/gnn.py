# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 19:00:57 2021

@author: peiji
"""

from spektral.datasets import TUDataset
from spektral.transforms import GCNFilter

dataset = TUDataset('PROTEINS')
dataset[0]
max_degree = dataset.map(lambda g: g.a.sum(-1).max(), reduce=max)

dataset.apply(GCNFilter())

#%%
 
# Creating GNN

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import BatchLoader

class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)
        return out
    
model = MyFirstGNN(32, dataset.n_labels)
model.compile('adam', 'categorical_crossentropy')



loader = BatchLoader(dataset, batch_size=32) # train set

model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=100)


loader = BatchLoader(dataset, batch_size=32) # test set

loss = model.evaluate(loader.load(), steps=loader.steps_per_epoch)

print('Test loss: {}'.format(loss))