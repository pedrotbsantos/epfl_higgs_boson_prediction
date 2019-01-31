# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
import csv

def load_data_csv(dataset_path):
    
    train_data = np.genfromtxt(dataset_path,skip_header=1 ,delimiter=',')
    higgs_feature = np.genfromtxt(dataset_path, delimiter=",", usecols=1 ,skip_header=1, dtype=str)
    for i in range(train_data.shape[0]):
        if higgs_feature[i] == 's':
            train_data[i,1] = 0
        else:
            train_data[i,1] = 1
    return train_data

def merge_data(index0, index1, index2, index3, pred0, pred1, pred2, pred3):
    data0 = np.c_[index0,pred0]
    data1 = np.c_[index1,pred1]
    data2 = np.c_[index2,pred2]
    data3 = np.c_[index3,pred3]
    
    merged_data = np.r_[data0,data1,data2,data3]
    merged_data = np.asarray(sorted(merged_data,key=lambda merged_data: merged_data[0]))
    return merged_data

def save_ws(w0,w1,w2,w3):
    np.save("w0.npy",w0)
    np.save("w1.npy",w1)
    np.save("w2.npy",w2)
    np.save("w3.npy",w3)

def load_ws():
    w0 = np.load("w0.npy")
    w1 = np.load("w1.npy")
    w2 = np.load("w2.npy")
    w3 = np.load("w3.npy")
    return w0,w1,w2,w3

def save_train_datasets(train_data_0,train_data_1,train_data_2,train_data_3):
    np.save("train_data_0.npy",train_data_0)
    np.save("train_data_1.npy",train_data_1)
    np.save("train_data_2.npy",train_data_2)
    np.save("train_data_3.npy",train_data_3)
    
def load_train_datasets():
    tr0 = np.load("train_data_0.npy")
    tr1 = np.load("train_data_1.npy")
    tr2 = np.load("train_data_2.npy")
    tr3 = np.load("train_data_3.npy")
    return tr0,tr1,tr2,tr3

def save_original_train_data(original_train_data):
    np.save("original_train_data.npy",original_train_data)
    
def load_original_train_data():
    otd = np.load("original_train_data.npy")
    return otd

    
def save_all():
    save_ws(w0,w1,w2,w3)
    save_train_datasets(train_data_0,train_data_1,train_data_2,train_data_3)
    save_original_train_data(original_train_data)


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
       
