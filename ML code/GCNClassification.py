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
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.utils import to_categorical

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
    """
    Convert values less than the threshold value to 0

    Parameters
    ----------
    W : 2D array, connevtivity matrix to be thresholded.
    p : float value between 0 and 1, Cell Value less than threshold value will be set to 0.
    copy : boolean, optional, The default is True.

    Raises
    ------
    ValueError, If the threshold is not within 0 and 1.

    Returns
    -------
    W : Thresholded 2D array, A matrix that does not contains negative values.

    """
    if p >= 1 or p <= 0:
        raise ValueError("Threshold value should be between 0 and 1")
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

    Parameters
    ----------
    adj_m : 2D array to be converted to adjacency list.

    Raises
    ------
    ValueError
        if connectivity matrix has length 0.

    Returns
    -------
    d : DataFrame.

    """
    # find non-zero elements in adj_mat
    if (len(adj_m) == 0):
        raise ValueError("Invalid adjacency matrix")

    indices = np.argwhere(adj_m)
    src, dsts = indices[:,0].reshape(-1, 1),indices[:,1].reshape(-1, 1)
    v = adj_m[src,dsts].reshape(-1, 1)
    final = np.concatenate((src, dsts, v), axis=1)
    d = pd.DataFrame(final)
    d.columns = ['source', 'target', 'weight']
    return d

def build_graphs(node_feat,adj_data):
    """
    Convert adjacency list to graphs

    Parameters
    ----------
    node_feat : 2D array
        where each row represent a node features for the brain region.
    adj_data : 2D array
        adjacency matrix.

    Raises
    ------
    ValueError
        if all the nodes are not connected.

    Returns
    -------
    graphs : Stellargraph Object
        to be used as an input later to the GNN model.

    """
    graphs = []
    min_T = np.min([item.shape[1] for item in node_feat])#assuming last dim is Time
    for A,X in zip(adj_data,node_feat):
        A_thr = threshold_proportional(A, 0.25)# adjacency matrix
        np.fill_diagonal(A_thr,1) # add selve-connectins to avoid zero in-degree nodes
        if np.sum(A_thr, axis=0).all() == False:
            raise ValueError("All the nodes are not connected")
        A_df = conv2list(A_thr)
        timeseries = X[:min_T]# node features (ROI,Time)
        X_df = pd.DataFrame(timeseries)
        G = StellarGraph(X_df, A_df)
        graphs.append(G)
    return graphs
        

def load_data():
    with open(args.input_data,'rb') as f:
        conn_data = pickle.load(f)
    conn_matrices = conn_data['FC']
    labels = conn_data['labels']
    graphs = build_graphs(conn_matrices,conn_matrices)
    return graphs, pd.Series(labels)

def create_model(generator):
    gc_model = GCNSupervisedGraphClassification(
         layer_sizes=[100, 100],
         activations=["relu", "relu"],
         generator=generator,
         dropout=0.5
         
         )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = BatchNormalization(momentum=0.9)
    predictions = Flatten()(x_out)
    predictions = Dense(units=64, activation="relu")(predictions)
    predictions = Dense(units=32, activation="relu")(predictions)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', 
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
    # num_classes = len(np.unique(train_subjects.values))
    # train_targets = to_categorical(train_subjects.values,num_classes)
    # val_targets = to_categorical(val_subjects.values,num_classes)
    # test_targets = to_categorical(test_subjects.values,num_classes)
    
    train_targets = train_subjects.values
    val_targets = val_subjects.values
    test_targets = test_subjects.values

    
    # Prepare graph generator
    generator = PaddedGraphGenerator(graphs=graphs)#FullBatchNodeGenerator(G, method="gat", sparse=False)
    train_gen = generator.flow(train_subjects.index, train_targets,batch_size=32)
    val_gen = generator.flow(val_subjects.index, val_targets,batch_size=32)
    test_gen = generator.flow(test_subjects.index, test_targets,batch_size=1)
    
    
     #train model
    model = create_model(generator)
    
    _info('Train GCN model')
    
    filename = MODEL_DIR + 'best_model.hdf5'
    checkpointer = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=1,  
                                   save_best_only=True, save_weights_only=True)
    
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=1,
            mode='auto', baseline=None, restore_best_weights=False)
    
    hist = model.fit(
            train_gen, validation_data=val_gen, shuffle=False, epochs=100, verbose=1,
            callbacks=[checkpointer,earlystopping])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(hist.history['accuracy'], label='train accuracy', color='green', marker="o")
    ax1.plot(hist.history['val_accuracy'], label='valid accuracy', color='blue', marker = "v")
    ax2.plot(hist.history['loss'], label = 'train loss', color='orange', marker="o")
    ax2.plot(hist.history['val_loss'], label = 'valid loss', color='red', marker = "v")
    ax1.legend(loc=3)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Loss', color='b')
    ax2.legend(loc=4)
    plt.ylim([0.6, 2.5])
    plt.show()
    
    model.save("models/gcn")
    
     #prediction
    _info('Test GCN model')
    del model
    
    # generator = PaddedGraphGenerator(graphs=graphs)
    # test_gen=generator.flow([0], [0])
    
    # saved_model = tf.keras.models.load_model("model/gcn_model")
    # # saved_model.summary()
    # predictions=saved_model.predict(test_gen, verbose=0).squeeze()
    # print("Predictions = {:.2}".format(predictions))
    
    generator = PaddedGraphGenerator(graphs=graphs)#FullBatchNodeGenerator(G, method="gat", sparse=False)
    model = create_model(generator)
    model.load_weights(filename)
    predictions = model.predict(test_gen,verbose=0).squeeze()
    # y_pred = np.argmax(predictions, 1)
    
    y_true = test_subjects.values
    
    y_pred = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
    
    Acc = accuracy_score(y_true,y_pred)
    Pre = precision_score(y_true,y_pred) 
    Rec = recall_score(y_true,y_pred)
    F1 = f1_score(y_true,y_pred)
    ROC = roc_auc_score(y_true,y_pred)
    
    
    _info('Print Results')
    print('Accuracy  = {:.2%}'.format(Acc))
    print('Precision = {:.2%}'.format(Pre))
    print('Recall    = {:.2%}'.format(Rec))
    print('F1_score  = {:.2%}'.format(F1))
    print('ROC_AUC  = {:.2%}'.format(ROC))
    
    
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')

    # data parameters
    parser.add_argument('-d', '--input-data', type=str,
        default='data/connectivity_matrices.pkl', help='path/to/roi/data')
    
    
    args = parser.parse_args()
    
    # load_data()
    # run()
    
    with open(args.input_data,'rb') as f:
        conn_data = pickle.load(f)
    tangent_matrices = conn_data['FC']
    labels = conn_data['labels']
    
    graphs = build_graphs([tangent_matrices[0]],[tangent_matrices[0]])
    labels = pd.Series(labels[0])

    generator = PaddedGraphGenerator(graphs=graphs)
    test_gen=generator.flow([0], labels)
    
    saved_model = tf.keras.models.load_model("models/gcn")
    
    predictions=saved_model.predict(test_gen, verbose=0).squeeze()
    print("Prediction class = {:.2}".format(predictions))
    print('finished!')
