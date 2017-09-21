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
from sklearn.cross_validation import train_test_split

# Here you specify the iterations. Lower it to test initially.
TOTAL_OUTER_ITERATIONS = 20 # Outer iteration reshuffles the 1000 non hab.
TOTAL_INNER_ITERATIONS = 100 # Inner iteration resplits the train and test sets.

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

algorithms = {
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forests': RandomForestClassifier()
}

#Retrieving data from PHL-HEC
data_object = retrieveHECData.HECDataFrame()
data_object.populatePreprocessedData()

for algo, clf in algorithms.items():
	print('Testing', algo)
	iter_count = 0
	for outer_iter in range(TOTAL_OUTER_ITERATIONS):
		data_non_hab, data_psychro, data_meso = data_object.returnSubsamples()
		
		for inner_iter in range(TOTAL_INNER_ITERATIONS):
			
			#Creating sets of class NON-HABITABLE
			train_non_hab, test_non_hab = train_test_split(data_non_hab, test_size=0.2)
			train_non_hab_labels = [1 for x in range(len(train_non_hab))]
			test_non_hab_labels = [1 for x in range(len(test_non_hab))]
			
			#Creating sets of class PSYCHROPLANET
			train_psychro, test_psychro = train_test_split(data_psychro, test_size=0.2)
			train_psychro_labels = [2 for x in range(len(train_psychro))]
			test_psychro_labels = [2 for x in range(len(test_psychro))]
			
			#Creating sets of class MESOPLANET
			train_meso, test_meso = train_test_split(data_meso, test_size=0.2)
			train_meso_labels = [3 for x in range(len(train_meso))]
			test_meso_labels = [3 for x in range(len(test_meso))]
			
			#Creating training and testing sets
			training_set = pd.concat([train_non_hab, train_psychro, train_meso])
			training_set = training_set.values
			test_set = pd.concat([test_non_hab, test_psychro, test_meso])
			test_set = test_set.values
			
			
			#Creating training and testing labels
			training_labels = train_non_hab_labels + train_psychro_labels + train_meso_labels
			test_labels = test_non_hab_labels + test_psychro_labels + test_meso_labels
			
			#Building classifiers
			try:
				clf.fit(training_set, training_labels)
				predicted_labels = clf.predict(test_set)
				print(test_labels)
				print(predicted_labels)
				print('-----')
			except:
				pass

