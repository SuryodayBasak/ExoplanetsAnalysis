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
#from xgboost.sklearn import XGBClassifier
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
data_object = retrieveHECData.HECDataFrame(download_new_flag = 0)
data_object.populatePreprocessedData()

data_non_hab, data_psychro, data_meso = data_object.returnAllSamples()

frames = [data_non_hab, data_psychro, data_meso]
df = pd.concat(frames)

feats1 = ['P. Min Mass (EU)', 'P. Esc Vel (EU)', 'P. Mean Distance (AU)', 'S. Mass (SU)', 'S. Radius (SU)', 'S. Teff (K)', 'S. Luminosity (SU)', 'P. Ts Mean (K)']
feats2 = ['P. Min Mass', 'P. Max Mass', 'P. Esc Vel', 'P. Mean Distance', 'P. Ts Mean (K)']

df_feats = df[feats1]

#temp_vals = list(df_feats['P. Ts Mean (K)'])
#df_feats = df_feats.drop(['P. Ts Mean (K)'], axis=1)

train, test = train_test_split(df_feats, test_size=0.2)
train_lbls = list(train['P. Ts Mean (K)'])
test_lbls = list(test['P. Ts Mean (K)'])
train = train.drop(['P. Ts Mean (K)'], axis = 1)
test = test.drop(['P. Ts Mean (K)'], axis = 1)

print(test)
