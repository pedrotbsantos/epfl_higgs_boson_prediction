import numpy as np


def check_dependencies(train_data):
    err_5 = 0
    err_6 = 0
    err_7 = 0
    err_8 = 0
    num_rows = train_data.shape[0]
    for i in range(num_rows):
        if (train_data[i,6] != -999.0) and (train_data[i,25] < 2):
            err_5 = 1
        if train_data[i,7] != -999.0 and train_data[i,25] < 2:
            err_6 = 1
        if train_data[i,8] != -999.0 and train_data[i,25] < 2:
            err_7 = 1
        if train_data[i,14] != -999.0 and train_data[i,25] < 2:
            err_8 = 1
    print(err_5)
    print(err_6)
    print(err_7)
    print(err_8)
  
    
def split_data(train_data):
    
    num_cols = train_data.shape[1]
    num_rows = train_data.shape[0]

    train_data = np.asarray(sorted(train_data,key=lambda train_data: train_data[24]))
    index_0 = 0
    index_1 = 0
    index_2 = 0
    index_3 = num_rows

    i = 0
    for i in range(0,num_rows):                     
        if (train_data[i][24] == 1 and index_0 == 0):
            index_0 = i
            
        if (train_data[i][24] == 2 and index_1 == 0):
            index_1 = i
            
        if (train_data[i][24] == 3 and index_2 == 0):
            index_2 = i
            
           
    train_data_0 = train_data[:index_0]
    train_data_0 = np.delete(train_data_0, [24], axis=1)
    
    train_data_1 = train_data[index_0:index_1]
    train_data_1 = np.delete(train_data_1, [24], axis=1)

    train_data_2 = train_data[index_1:index_2]
    train_data_2 = np.delete(train_data_2, [24], axis=1)

    train_data_3 = train_data[index_2:index_3]
    train_data_3 = np.delete(train_data_3, [24], axis=1)
    
    
    return train_data_0, train_data_1, train_data_2, train_data_3, index_0, index_1, index_2, index_3


def stat_cols(train_data, missing_parameter):
    
    num_cols = train_data.shape[1]
    num_rows = train_data.shape[0]
    stats = np.zeros(shape=(num_cols,2))
    
    for i in range(2,num_cols):                
        buffer = 0
        for j in range(0,num_rows):            
            if (train_data[j,i] == missing_parameter):   #-999.0
                buffer += 1
        stats[i] = [i+1,buffer/num_rows]
    
    
    return stats


def delete_rows(train_data, stats):
    
    num_rows_stats = stats.shape[0]
    i = 0
    while i < num_rows_stats:
        if stats[i,1] == 1. : 
            train_data = np.delete(train_data, i,1)  #delete i column
            stats = np.delete(stats, i,0)            #delete i row
            i -= 1                                   #if we delete a column we have to check it next time again
            
        if stats[i,0] == num_rows_stats:             #out of while loop
            i = num_rows_stats  
        
        i += 1    
    
    print(train_data.shape)
    return train_data

def mean_replacement(train_data, column):
    num_rows_train = train_data.shape[0]
    sum_data = 0
    counter = 0
    
    #calculate mean, based on valid data points
    for i in range(num_rows_train):
        if train_data[i,column] != -999.0:
            sum_data += train_data[i,column]
            counter += 1
    mean = sum_data/counter

    #replace missing parameters with mean
    for i in range(num_rows_train):
        if train_data[i,column] == -999.0:
            train_data[i,column] = mean
    
    return train_data

def execute_mean_replacement(train_data, stats):
    num_rows = stats.shape[0]
    mean_th = 0.3         
    
    for i in range(num_rows):
        if stats[i,1] < mean_th:
            train_data = mean_replacement(train_data, i)
    
    return train_data

def standardize_data(train_data):
    num_cols  = train_data.shape[1]
    train_data_original = train_data
    train_data = np.delete(train_data, 0 ,1)
    train_data = np.delete(train_data, 0 ,1)
    
    mean_x = np.mean(train_data)
    train_data = train_data - mean_x
    std_x = np.std(train_data)
    train_data = train_data / std_x
    
    
    train_data = np.c_[train_data_original[:,1], train_data]
    train_data = np.c_[train_data_original[:,0], train_data]
    return train_data


