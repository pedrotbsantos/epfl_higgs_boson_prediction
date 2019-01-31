# -*- coding: utf-8 -*-
"""
Implementations of the 6 functions needed
"""
import numpy as np
from helpers import batch_iter

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma * grad
    loss = compute_mse(y,tx,w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx, batch_size=1, num_batches=1):
            grad = compute_gradient(y_batch,tx_batch,w)
            w = w - gamma * grad
    loss = compute_mse(y,tx,w)
    return w, loss

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y,tx,w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    a = tx.T.dot(tx) + lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w, lambda_)
    return w, loss

def logistic_regressionn(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx, batch_size=1, num_batches=1):
            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx, batch_size=1, num_batches=1):
            grad = compute_logistic_gradient(y_batch, tx_batch, w, lambda_)
            w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w, lambda_)
    return w, loss


#needed in the function above
def compute_mse(y, tx, w, lambda_=0):
    e = y - tx.dot(w)
    N = y.shape[0]
    norm = np.linalg.norm(w)
    mse = (1/(2*N))*e.T.dot(e)
    return mse + lambda_ * norm

def compute_logistic_loss(y, tx, w, lambda_=0):
    N = y.shape[0]
    loss = 0
    for n in range(N):
        ln = np.log(1 + np.exp(tx[n].T.dot(w)))
        loss += ln - y[n]*(tx[n].T.dot(w))
    norm = np.linalg.norm(w)
    return loss + (lambda_/2) * norm

def compute_gradient(y, tx, w):
    N = len(y)
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / N
    return grad

def compute_logistic_gradient(y, tx, w, lambda_=0):
    sigm_txw = sigmoid(tx.dot(w.T))
    e = sigm_txw - y
    grad = e.dot(tx) + lambda_*w
    return grad

def sigmoid(x):
    e_minusX = np.exp(-x)
    return 1 / (1 + e_minusX)
