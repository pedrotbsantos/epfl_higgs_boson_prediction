# -*- coding: utf-8 -*-
import numpy as np
from implementations import *


#---------------------------------------- COMPUTE MODEL ----------------------------------------

def compute_model_ridge(x, y, lambdas, degrees ,seeds=range(1)):
    # number of columns used
    columns = x.shape[1]
    
    # parameters
    k_fold = 4   
    
    best_w = np.empty(columns+1)
    lowest_loss = 999999999
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
        k_indices = build_k_indices(y, k_fold, seed)
        for index, i in enumerate(degrees):    
            #testing different degrees
            print("------Degree "+ str(i))
            #poly_i = expand_features_with_crossterms(x, i)
            poly_i = feat_exp(x, i)

            for index_lambda, lambda_ in enumerate(lambdas):

                print("lambda: " + str(index_lambda) + "/" + str(len(lambdas)))
             
                loss_te = 0
                for j in range(k_fold):
                    w, _, loss_te_j = cross_validation_ridge_regression(y, k_indices, j, lambda_, i, poly_i)

                    
                    loss_te += loss_te_j
        
                loss_te /= k_fold
                
                if (loss_te < lowest_loss):
                    best_deg = i
                    best_w = w
                    best_poly = poly_i
                    lowest_loss = loss_te
    return best_w, lowest_loss, best_poly, best_deg

#---------------------------------------- CROSS VALIDATION ----------------------------------------  


def cross_validation_ridge_regression(y, k_indices, k, lambda_, degree, poly):
    """return the loss of ridge regression."""
    train_indices, test_indices = train_test_indices(k_indices, k)
    
    test_y = y[k_indices[k]]
    train_y = np.delete(y, k_indices[k])
    
    test_x = poly[k_indices[k],:]
    train_x = np.delete(poly, k_indices[k], 0)
    
    
    w, loss = ridge_regression(train_y, train_x, lambda_)
    
    rmse_tr = np.sqrt(2 * loss)
    rmse_te = np.sqrt(2 * compute_mse(test_y, test_x, w))
    return w, rmse_tr, rmse_te




def cross_validation_reg_logistic_regression(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """return the loss of ridge regression."""
    train_indices, test_indices = train_test_indices(k_indices, k)
    
    train_x = x[train_indices]
    train_y = y[train_indices]
    test_x = x[test_indices]
    test_y = y[test_indices]
    
    w, loss = reg_logistic_regression(train_y, train_x, lambda_, initial_w, max_iters, gamma)
    
    rmse_tr = np.sqrt(2 * loss)
    rmse_te = np.sqrt(2 * compute_logistic_loss(test_y, test_x, w))
    return w, rmse_tr, rmse_te

#---------------------------------------- FEATURE EXPANDTION ----------------------------------------  

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def expand_features_with_crossterms(x, degree):
    l = x.shape[1]
    elem = x.shape[0]
    x_prev = x.copy()
    ind_prev = np.arange(l+1)
    ind_new = ind_prev.copy()
    for i in range(1,degree):
        x_exp = np.empty((elem,int(x_prev.shape[1]*(l+i)/(i+1))))
        for n in range(l):
            for m in range(ind_prev[n],ind_prev[l]):
                x_exp[:,ind_new[n]+m-ind_prev[n]] = x[:,n]*x_prev[:,m]
            ind_new[n+1] = ind_prev[l] - ind_prev[n] + ind_new[n]
        x = np.c_[(x,x_exp)]
        x_prev = x_exp.copy()
        ind_prev = ind_new.copy()
    return x

def feat_exp(x,degrees):
    
    x = np.c_[np.ones(x.shape[0]),x]
    poly_i = build_poly(x, degrees)
        
    return poly_i


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def train_test_indices(k_indices, k):
    test_indices = k_indices[k]
    train_indices = np.concatenate((k_indices[:k],k_indices[k+1:]),axis=None)
    return train_indices, test_indices


#---------------------------------------- PREDICTION ----------------------------------------  

def check_best_perc(sorted_train_data, poly, w, min_value, max_value, top, bottom, index):
    
    num_rows_t = sorted_train_data.shape[0]
    num_rows_p = poly.shape[0]
    best_perc = min_value
    best_error = 100
    for perc in range(int(min_value*2000), int(max_value*2000)):
        pred = predict(poly,w,perc/2000, index)
        errors, pers_error = number_errors(sorted_train_data[top:bottom,1],pred)
        if pers_error < best_error:
            best_error = pers_error
            best_perc = perc/2000
    return best_error, best_perc

def execute_best_perc(sorted_train_data, poly0, poly1, poly2, poly3, w0, w1, w2, w3, index): 
    num_rows_t = sorted_train_data.shape[0]
    num_rows_p0 = poly0.shape[0]
    num_rows_p1 = poly1.shape[0]
    num_rows_p2 = poly2.shape[0]
    num_rows_p3 = poly3.shape[0]

    index_p0_t = 0
    index_p0_b = num_rows_p0
    index_p1_t = index_p0_b 
    index_p1_b = num_rows_p0 + num_rows_p1
    index_p2_t = index_p1_b 
    index_p2_b = num_rows_p0 + num_rows_p1 + num_rows_p2
    index_p3_t = index_p2_b 
    index_p3_b = num_rows_p0 + num_rows_p1 + num_rows_p2 + num_rows_p3
  
    print("--------Percentage--------")

    best_error0, best_perc0 = check_best_perc(sorted_train_data ,poly0, w0, 0.5,0.7, index_p0_t,index_p0_b, index)
    best_error1, best_perc1 = check_best_perc(sorted_train_data, poly1, w1, 0.5,0.7, index_p1_t,index_p1_b, index)
    best_error2, best_perc2 = check_best_perc(sorted_train_data ,poly2, w2, 0.5,0.7, index_p2_t,index_p2_b, index)
    best_error3, best_perc3 = check_best_perc(sorted_train_data, poly3, w3, 0.5,0.7, index_p3_t,index_p3_b, index)
    
    print("TrainSet0:    " + str(best_error0) + "    with perc_value:   " + str(best_perc0))
    print("TrainSet1:    " + str(best_error1) + "    with perc_value:   " + str(best_perc1))
    print("TrainSet2:    " + str(best_error2) + "    with perc_value:   " + str(best_perc2))
    print("TrainSet3:    " + str(best_error3) + "    with perc_value:   " + str(best_perc3))

    avg_result = (best_error0*num_rows_p0 + best_error1*num_rows_p1 + best_error2*num_rows_p2 + best_error3*num_rows_p3)/num_rows_t 
    print("Average Value: " + str(avg_result))
    
    return best_perc0, best_perc1, best_perc2, best_perc3

def predict(x,w,p,index):
    sigm_xw = sigmoid(x.dot(w))
    y_pred = np.empty(sigm_xw.shape[0])
    for i in range(sigm_xw.shape[0]):
        if (sigm_xw[i] < p):
            if index == 0:       #index 0 --> test Dataset
                y_pred[i] = 1
            if index == 1:       #index 1 --> train Dataset
                y_pred[i] = 0
        else:
            if index == 0:
                y_pred[i] = -1
            if index == 1:
                y_pred[i] = 1
    return y_pred



def execute_standard_prediction(poly0, poly1, poly2, poly3, w0, w1, w2, w3, index):
    perc = 0.62

    pred0 = predict(poly0,w0,perc, index)
    pred1 = predict(poly1,w1,perc, index)
    pred2 = predict(poly2,w2,perc, index)
    pred3 = predict(poly3,w3,perc, index)
    
    return pred0, pred1, pred2, pred3

def execute_advanced_prediction(poly0, poly1, poly2, poly3, w0, w1, w2, w3, sorted_train_data, index):
    
    perc0, perc1, perc2, perc3 = execute_best_perc(sorted_train_data, poly0, poly1, poly2, poly3, w0, w1, w2, w3, index)
    
    pred0 = predict(poly0,w0,perc0, index)
    pred1 = predict(poly1,w1,perc1, index)
    pred2 = predict(poly2,w2,perc2, index)
    pred3 = predict(poly3,w3,perc3, index)
    
    return pred0, pred1, pred2, pred3

#---------------------------------------- CHECK TRAINING PREDICTION ----------------------------------------  

def number_errors(y,pred_y):
    errors = 0
    for i in range(len(y)):
        if (pred_y[i] != y[i]):
            errors += 1
    return errors, 100*errors/len(y)