import sys
import csv
import numpy as np
import pandas as pd
import retrieveHECData
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


np.set_printoptions(precision=2)

# Here is the list of algorithms we will use.
algorithms = {
    #'Gaussian Naive Bayes': GaussianNB(),
    #'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    #'Support Vector Machine': SVC(kernel='linear'),
    #'Radial Basis SVM': SVC(kernel='rbf'),
    #'K Nearest Neighbors': KNeighborsClassifier(),
    #'Decision Trees': DecisionTreeClassifier(),
    'Random Forests': RandomForestClassifier(n_estimators=1000)
}

# Here we specify the iterations. Lower it to test initially.
O_ITER = 10 # Outer iteration reshuffles the 1000 non hab.
I_ITER = 10 # Inner iteration resplits the train and test sets.

#Retrieving data from PHL-HEC
data_object = retrieveHECData.HECDataFrame(download_new_flag = 0)
data_object.populatePreprocessedData()
data_nh, data_p, data_m = data_object.returnAllSamples()

#confusion matrix initialized

for algo, clf in algorithms.items():
    #Generating the train labels
    data_lbl_nh = [0 for x in range(len(data_nh))]
    data_lbl_p = [1 for x in range(len(data_p))]
    data_lbl_m = [2 for x in range(len(data_m))]

    #Generating training set and labels
    train_frames = [data_nh, data_p, data_m]
    train_x = pd.concat(train_frames, ignore_index=True)
    train_y = data_lbl_nh + data_lbl_p + data_lbl_m

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
    #test_x = test_x.values

    xgb_train = xgb.DMatrix(train_x, train_y)
    #xgb_test = xgb.DMatrix(test_x)
    bst = xgb.train(param, xgb_train, num_round)
    #pred_labels = bst.predict(xgb_test)
    importances = bst.feature_importances_

    feature_names = list(data_nh)
    #bst.booster().get_score(importance_type='weight')
    #print(get_xgb_imp(bst, feature_names))
    #importance = bst.get_fscore(fmap=feature_names)
    #importance = sorted(importance.items(), key=operator.itemgetter(1))
    #print(importance)

    for val in indices:
        print(feature_names[val], '\\\\')
