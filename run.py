
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from data_cleaning_helpers import *
from model import *

#######################################################################################
#
#      for explanations please take a look at the following notebooks:
#
#           DATA CLEANING:          "Data Cleaning.ipynd"
#           MODELING:              "Computing Model.ipynd"
#           TESTING:          "Apply model on Test Data.ipynd"
#
#     this run.py file is a copy of the important parts of the "TESTING" notebook
#
#######################################################################################

# Please change the dataset_path to the directory of the testing data
dataset_path = 'Data/test.csv'

test_data = load_data_csv(dataset_path)
[w0,w1,w2,w3] = load_ws()

test_data_0, test_data_1, test_data_2, test_data_3, index_0, index_1, index_2, index_3 = split_data(test_data)

stats_0 = stat_cols(test_data_0, -999.0)
stats_1 = stat_cols(test_data_1, -999.0)
stats_2 = stat_cols(test_data_2, -999.0)
stats_3 = stat_cols(test_data_3, -999.0)
stats_original = stat_cols(test_data, -999.0)

test_data_0 = delete_rows(test_data_0, stats_0)
test_data_1 = delete_rows(test_data_1, stats_1)
test_data_2 = delete_rows(test_data_2, stats_2)
test_data_3 = delete_rows(test_data_3, stats_3)

stats_0 = stat_cols(test_data_0, -999.0)
stats_1 = stat_cols(test_data_1, -999.0)
stats_2 = stat_cols(test_data_2, -999.0)
stats_3 = stat_cols(test_data_3, -999.0)

test_data_0 = execute_mean_replacement(test_data_0, stats_0)
test_data_1 = execute_mean_replacement(test_data_1, stats_1)
test_data_2 = execute_mean_replacement(test_data_2, stats_2)
test_data_3 = execute_mean_replacement(test_data_3, stats_3)

test_data_0 = standardize_data(test_data_0)
test_data_1 = standardize_data(test_data_1)
test_data_2 = standardize_data(test_data_2)
test_data_3 = standardize_data(test_data_3)

stats_0 = stat_cols(test_data_0, -999.0)
stats_1 = stat_cols(test_data_1, -999.0)
stats_2 = stat_cols(test_data_2, -999.0)
stats_3 = stat_cols(test_data_3, -999.0)

x0 = test_data_0[:,2:]
x1 = test_data_1[:,2:]
x2 = test_data_2[:,2:]
x3 = test_data_3[:,2:]
index0 = test_data_0[:,0]
index1 = test_data_1[:,0]
index2 = test_data_2[:,0]
index3 = test_data_3[:,0]

poly0 = feat_exp(x0,2)
poly1 = feat_exp(x1,5)
poly2 = feat_exp(x2,2)
poly3 = feat_exp(x3,5)

pred0 = predict(poly0,w0,0.627, 0)
pred1 = predict(poly1,w1,0.624, 0)
pred2 = predict(poly2,w2,0.6235, 0)
pred3 = predict(poly3,w3,0.628, 0)

merged_data = merge_data(index0, index1, index2, index3, pred0, pred1, pred2, pred3)

create_csv_submission(merged_data[:,0], merged_data[:,1], 'test_pred.csv')







