#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:43:06 2021

@author: student
"""
import pandas as pd
import numpy as np
import argparse
import os
import random
import matplotlib.pyplot as plt

from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# results directory
RES_DIR = 'results/gcn'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
MODEL_DIR = 'models/gcn/'
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 5000

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def _info(s):
    print('---')
    print(s)
    print('---')

def threshold_proportional(W, p, copy=True):
    assert p < 1 or p > 0
    if copy:
        W = W.copy()
    n = len(W)                        # number of nodes
    np.fill_diagonal(W, 0)            # clear diagonal
    if np.all(W == W.T):              # if symmetric matrix
        W[np.tril_indices(n)] = 0     # ensure symmetry is preserved
        ud = 2                        # halve number of removed links
    else:
        ud = 1
    ind = np.where(W)                    # find all links
    I = np.argsort(W[ind])[::-1]         # sort indices by magnitude
    # number of links to be preserved
    en = round((n * n - n) * p / ud)
    W[(ind[0][I][en:], ind[1][I][en:])] = 0    # apply threshold
    if ud == 2:                                # if symmetric matrix
        W[:, :] = W + W.T                      # reconstruct symmetry
    
    W[W>0.9999] = 1                            # make sure the highest correlation coeff is 1
    return W

def conv2list(adj_m):
        """
        converts adjacency matrix to adj list to load into stellargraph
        """
        
        # find non-zero elements in adj_mat
        indices = np.argwhere(adj_m)
        src, dsts = indices[:,0].reshape(-1, 1),indices[:,1].reshape(-1, 1)
        v = adj_m[src,dsts].reshape(-1, 1)
        final = np.concatenate((src, dsts, v), axis=1)
        d = pd.DataFrame(final)
        d.columns = ['source', 'target', 'weight']
        
            
        return d

def build_graphs(node_feat,adj_data):
    graphs = []
    min_T = np.min([item.shape[1] for item in node_feat])#assuming last dim is Time
    for A,X in zip(adj_data,node_feat):
        A_thr = threshold_proportional(A, 0.25)# adjacency matrix
        A_df = conv2list(A_thr)
        timeseries = X[:min_T]# node features (ROI,Time)
        X_df = pd.DataFrame(timeseries)
        G = StellarGraph(X_df, A_df)
        graphs.append(G)
        
    return graphs
        

def load_data():
    with open(args.input_data,'rb') as f:
        conn_data = pickle.load(f)
    tangent_matrices = conn_data['FC']
    labels = conn_data['labels']
    graphs = build_graphs(tangent_matrices,tangent_matrices)
    return graphs, pd.Series(labels)

def create_model(generator):
    gc_model = GCNSupervisedGraphClassification(
         layer_sizes=[111, 111],
         activations=["relu", "relu"],
         generator=generator,
         dropout=0.5
         )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy', 
                 metrics=["accuracy"])
    plot_model(model, to_file=MODEL_DIR+'gcn_model.png', show_shapes=True)
    return model

def run():
    graphs, graph_labels = load_data()
    # print(graphs[0].info())
    # print(graphs[1].info())
    # summary = pd.DataFrame(
    #         [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    #         columns=["nodes", "edges"],
    #     )
    # summary.describe().round(1)
    graph_labels.value_counts().to_frame()
    # split data into train-val-tes
    train_subjects, test_subjects = model_selection.train_test_split(
                                    graph_labels, test_size=0.2, stratify=graph_labels,#IDs
                                    random_state=SEED)
    train_subjects, val_subjects = model_selection.train_test_split(
                        train_subjects, test_size=0.2, stratify=train_subjects,
                        random_state=SEED)
    
    # this are all np array
    train_targets = train_subjects.values
    val_targets = val_subjects.values
    test_targets = test_subjects.values

    
    # Prepare graph generator
    generator = PaddedGraphGenerator(graphs=graphs)#FullBatchNodeGenerator(G, method="gat", sparse=False)
    train_gen = generator.flow(train_subjects.index, train_targets,batch_size=32)
    val_gen = generator.flow(val_subjects.index, val_targets,batch_size=32)
    test_gen = generator.flow(test_subjects.index, test_targets)
    
     #train model
    model = create_model(generator)
    
    _info('Train GCN model')
    
    filename = MODEL_DIR + 'best_model.hdf5'
    checkpointer = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=1,  
                                   save_best_only=True, save_weights_only=True)
    
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=1,
            mode='auto', baseline=None, restore_best_weights=False)
    
    history = model.fit(
            train_gen, validation_data=val_gen, shuffle=False, epochs=100, verbose=1,
            callbacks=[checkpointer,earlystopping])

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    model.save("model/gcn_model")
    
     #prediction
    _info('Test GCN model')
    del model
    
    # ADRIAN convert connectivity matrices to graph 1st before predicting 
    # and graphs must be an arr, so graphs=[1 graph]
    generator = PaddedGraphGenerator(graphs=graphs)
    test_gen=generator.flow([0], [0])
    
    saved_model = tf.keras.models.load_model("model/gcn_model")
    saved_model.summary()
    predictions=saved_model.predict(test_gen, verbose=0).squeeze()
    print("Predictions = {:.2}".format(predictions))
    
    # generator = PaddedGraphGenerator(graphs=graphs)#FullBatchNodeGenerator(G, method="gat", sparse=False)
    # model = create_model(generator)
    # model.load_weights(filename)
    # predictions = model.predict(test_gen,verbose=0).squeeze()
    # y_pred = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
    
    # Acc = accuracy_score(test_targets,y_pred)
    # Pre = precision_score(test_targets,y_pred) 
    # Rec = recall_score(test_targets,y_pred)
    # F1 = f1_score(test_targets,y_pred)
    # ROC = roc_auc_score(test_targets,y_pred)
    
    
    # _info('Print Results')
    # print('Accuracy  = {:.2%}'.format(Acc))
    # print('Precision = {:.2%}'.format(Pre))
    # print('Recall    = {:.2%}'.format(Rec))
    # print('F1_score  = {:.2%}'.format(F1))
    # print('ROC_AUC  = {:.2%}'.format(ROC))
    
    
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')

    # data parameters
    parser.add_argument('-d', '--input-data', type=str,
        default='data/connectivity_matrices.pkl', help='path/to/roi/data')
    
    
    args = parser.parse_args()
    
    # load_data()
    run()

    print('finished!')