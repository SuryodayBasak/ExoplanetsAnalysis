import sys
import csv
import numpy as np
import pandas as pd
import retrieveHECData
import xgboost as xgb

np.set_printoptions(precision=2)

def conf_matrix_inc(mat, orig_lbls, pred_lbls):
    if len(orig_lbls) != len(pred_lbls):
        print('Shapes of lists of labels do not match')
        return 0

    for i in range(len(orig_lbls)):
        mat[orig_lbls[i], pred_lbls[i]] += 1

def conf_matrix_probs(mat):
    r, c = np.shape(mat)

    for i in range(r):
        r_sum = sum(mat[i,:])
        for j in range(c):
            mat[i, j] = (mat[i, j]/r_sum) * 100

    return mat

# Here we specify the iterations. Lower it to test initially.
O_ITER = 5 # Outer iteration reshuffles the 1000 non hab.
I_ITER = 5 # Inner iteration resplits the train and test sets.

#Retrieving data from PHL-HEC
data_object = retrieveHECData.HECDataFrame(download_new_flag = 0)
data_object.populatePreprocessedData()
data_nh, data_p, data_m = data_object.returnAllSamples()

#confusion matrix initialized
conf_mat = np.zeros((3, 3))

for i in range(0, O_ITER):
    split_idx = np.random.rand(len(data_nh)) < 0.05
    sample_nh = data_nh[split_idx]
    for j in range(0, I_ITER):

        #Indexes for splitting into training and testing sets
        split_nh = np.random.rand(len(sample_nh)) < 0.8
        split_p = np.random.rand(len(data_p)) < 0.8
        split_m = np.random.rand(len(data_m)) < 0.8

        #Extracting the training samples
        train_nh = sample_nh[split_nh]
        train_p = data_p[split_p]
        train_m = data_m[split_m]

        #Generating the training labels
        train_lbl_nh = [0 for x in range(len(train_nh))]
        train_lbl_p = [1 for x in range(len(train_p))]
        train_lbl_m = [2 for x in range(len(train_m))]

        #Extracting the test samples
        test_nh = sample_nh[~split_nh]
        test_p = data_p[~split_p]
        test_m = data_m[~split_m]

        #Generating the test labels
        test_lbl_nh = [0 for x in range(len(test_nh))]
        test_lbl_p = [1 for x in range(len(test_p))]
        test_lbl_m = [2 for x in range(len(test_m))]

        #Generating training set and labels
        train_frames = [train_nh, train_p, train_m]
        train_x = pd.concat(train_frames, ignore_index=True)
        train_y = train_lbl_nh + train_lbl_p + train_lbl_m

        #Generating test set and labels
        test_frames = [test_nh, test_p, test_m]
        test_x = pd.concat(test_frames, ignore_index=True)
        test_y = test_lbl_nh + test_lbl_p + test_lbl_m

        #print(train_x)
        param = {'learning_rate':0.1,
		'n_estimators':1000,
		'max_depth':8,
		'min_child_weight':1,
		'eta':0.1,
		'silent':1,
		'objective':'multi:softmax',
		'num_class':3,
		'subsample':1,
		'gamma':0}
        num_round = 5

        """EDIT THIS"""
        #BRING IN THE MACHINE LEARNING SWAG RIGHT HERE

        train_x = train_x.values
        test_x = test_x.values

        xgb_train = xgb.DMatrix(train_x, train_y)
        xgb_test = xgb.DMatrix(test_x)
        bst = xgb.train(param, xgb_train, num_round)
        pred_labels = bst.predict(xgb_test)

        #updating the confusion matrix
        conf_matrix_inc(conf_mat, test_y, list(pred_labels.astype(int)))

final_conf_mat = conf_matrix_probs(conf_mat)
final_conf_mat = np.round_(final_conf_mat, decimals = 2)
#LaTeX Dump
print(' & Non Habitable &',  final_conf_mat[0, 0],
                            '&', final_conf_mat[0, 1],
                            '&', final_conf_mat[0, 2],
                            '\\\\')

print('GBDT & Psychroplanets &',  final_conf_mat[1, 0],
                            '&', final_conf_mat[1, 1],
                            '&', final_conf_mat[1, 2],
                            '\\\\')

print(' & Mesoplanets &',  final_conf_mat[2, 0],
                            '&', final_conf_mat[2, 1],
                            '&', final_conf_mat[2, 2],
                            '\\\\')

print('\hline')
