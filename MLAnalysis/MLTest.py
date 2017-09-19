'''
classify.py - Run all algorithms on the habitability data set given
that nonhabitable planets are in a different file from the other
classes.

Inner iterations are parallelized across 4 workers. Keep TOTAL_INN to a
multiple of 4 since the actual number of iterations performed is,
    => (TOTAL_INN // 4) * 4

'''

import sys
import csv
import numpy as np
import pandas as pd
import retrieveHECData
from concurrent.futures import ProcessPoolExecutor

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix

# Here you specify the iterations. Lower it to test initially.
TOTAL_OUTER_ITERATIONS = 20          # Outer iteration reshuffles the 1000 non hab.
TOTAL_INNER_ITERATIONS = 100         # Inner iteration resplits the train and test sets.

"""
xgbparams = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 8,
    'min_child_weight': 1,
    'eta': 0.1,
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 3,
    'subsample': 1,
    'gamma': 0,
    'n_jobs': 1
}
"""
# Add the algorithms you want here. Set parameters as required. The key
# is used in a print statement so set whatever suits you.
algorithms = {
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forests': RandomForestClassifier(),
    #'XG Boost': XGBClassifier(**xgbparams)
}

#Retrieving data from PHL-HEC
data_object = retrieveHECData.HECDataFrame()
data_object.populatePreprocessedData()

for algo, clf in algorithms.items():
	print('Testing', algo)
	
	for outer_iter in range(TOTAL_OUTER_ITERATIONS):
		data_non_hab, data_psychro, data_meso = data_object.returnSubsamples()
		
		for inner_iter in range(TOTAL_INNER_ITERATIONS):
			pass
