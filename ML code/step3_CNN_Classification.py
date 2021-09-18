#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:07:07 2021

@author: student
"""


import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
# import time
'''
ml
'''
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
'''
Helpers
'''
def _info(s):
    print('---')
    print(s)
    print('---')


#%% create conv model
def create_model(data_shape_full, num_blocks):
    model = Sequential()
    for i in range(num_blocks):
        if i==0:
            model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',
                             strides=(1,1), activation='relu', input_shape=data_shape_full))
        else:
            model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                             strides=(1,1), activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',
                             strides=(1,1), activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.5))
    # regression layer with one output only
    model.add(Dense(1, activation='sigmoid'))# binary classification
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
          loss='binary_crossentropy', 
          metrics=['accuracy'])
    
    print(model.summary())
    return model

def train_valid_test_split(data, labels, train_size=0.6, test_size=0.2, rand_seed=33):
    valid_size = 1 - (train_size + test_size)
    X1, X_test, y1, y_test = train_test_split(data, labels, test_size = test_size, random_state= rand_seed)
    X_test= X_test
    y_test= y_test
    X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size = float(valid_size)/(valid_size+ train_size))
    X_train= X_train
    y_train= y_train
    X_valid= X_valid
    y_valid= y_valid
    return X_train, X_valid, X_test, y_train, y_valid, y_test

#%%
def run(args):   
    
    '''
    get functional connectivity data
    '''
    _info('Loading data ..')
    with open(args.input_data,'rb') as f:
       data = pickle.load(f)
       
    # fc_data = connectivity matrix
    fc_data, labels = data['FC'], data['labels']
    
    # # split data into train and test
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, test_idx = next(kf.split(fc_data,labels))
    
    X_train, y_train, X_test, y_test = \
        fc_data[train_idx], labels[train_idx], fc_data[test_idx], labels[test_idx]
    
    # X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(fc_data, labels)
    
    del data, fc_data, labels
    
    # build CNN model
    _info('Build CNN model')
    X_train = X_train[...,None]# add channel dimension for CNN
    # X_valid = X_valid[...,None]
    X_test  = X_test[...,None]
    dim = X_train.shape[1:]
    model = create_model(dim, 1)
    
    #train model
    _info('Train CNN model')
    saveModels = './cnn_models/'
    os.makedirs(saveModels, exist_ok=True)
    filename = saveModels + 'best_model.hdf5'
    checkpointer = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=1,  
                                   save_best_only=True)
    # early_checkpoint = EarlyStopping(patience=10, monitor='val_loss', mode='min')
    
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(x = X_train)

    it = datagen.flow(x = X_train, 
                      y = y_train, 
                      batch_size=32)
    
    hist = model.fit(x=X_train,
                     y=y_train,
                      epochs=100,
                      batch_size=32,
                      shuffle=True,
                      validation_split=0.1,
                      verbose = 1,
                      callbacks=[checkpointer])
    
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
    
    
    #prediction
    _info('Test CNN model')
    del model
    model = load_model(filename)
    # predictions = model.predict(X_test,verbose=0).squeeze()
    # y_pred = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
     
    # acc = accuracy_score(y_test, y_pred)
    
    # _info('Print Results')
    # print('Test Acccuracy = {:.2%}'.format(acc))
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1, batch_size=64)
    print("Test acc is {}".format(test_acc))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')

    # data parameters
    parser.add_argument('-d', '--input_data', type=str,
        default='data/connectivity_matrices.pkl', help='path/to/FC/data')

    
    args = parser.parse_args()

    # run(args)
    
    with open(args.input_data,'rb') as f:
        data = pickle.load(f)
       
    # fc_data = connectivity matrix
    fc_data, labels = data['FC'], data['labels']
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(fc_data, labels)
    X_train = X_train[...,None]
    X_valid = X_valid[...,None]
    X_test  = X_test[...,None]
    
    model = load_model('./cnn_models/best_model.hdf5')
    predictions = model.predict(X_test,verbose=0).squeeze()
    y_pred = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
    # predictions = model.predict(xtest,verbose=0).squeeze()
     
    acc = accuracy_score(y_test, y_pred)
    Pre = precision_score(y_test,y_pred) 
    Rec = recall_score(y_test,y_pred)
    F1 = f1_score(y_test,y_pred)
    ROC = roc_auc_score(y_test,y_pred)
    
    
    _info('Print Results of CNN model')
    print('Test Acccuracy = {:.2%}'.format(acc))
    print('Precision = {:.2%}'.format(Pre))
    print('Recall    = {:.2%}'.format(Rec))
    print('F1_score  = {:.2%}'.format(F1))
    print('ROC_AUC  = {:.2%}'.format(ROC))
    
    # test_loss, test_acc = model.evaluate(X_train,  y_train, verbose=1, batch_size=64)
    # print("Test acc is {}".format(test_acc))

    print('finished!')