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

class TestPlanets:
	def __init__(self):
		self.planet_names = []
		print('Collecting list of features.')
		with open('test-planets.txt') as fp:
			for line in fp:
				self.planet_names.append(line[:-1])

	def returnPlanetNames(self):
		return self.planet_names

# Here you specify the iterations. Lower it to test initially.
TOTAL_OUTER_ITERATIONS = 100 # Outer iteration reshuffles the 1000 non hab.
TOTAL_INNER_ITERATIONS = 10 # Inner iteration resplits the train and test sets.

algorithms = {
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forests': RandomForestClassifier()
}

#Retrieving data from PHL-HEC
data_object = retrieveHECData.HECDataFrame(download_new_flag = 0)
data_object.populatePreprocessedData()
test_planets = TestPlanets()
planet_names = test_planets.returnPlanetNames()
print(planet_names)

data_nh, data_p, data_m = data_object.returnAllSamples()
#print(data_nh)
#print('Searching in NH')
test_planets_nh = data_nh[data_nh['P. Name'].isin(planet_names)]
train_planets_nh = data_nh[~data_nh['P. Name'].isin(planet_names)]
#print(test_planets_nh)

#print('Searching in P')
test_planets_p = data_p[data_p['P. Name'].isin(planet_names)]
train_planets_p = data_p[~data_p['P. Name'].isin(planet_names)]
#print(test_planets_p)

#print('Searching in M')
test_planets_m = data_m[data_m['P. Name'].isin(planet_names)]
train_planets_m = data_m[~data_m['P. Name'].isin(planet_names)]
#print(test_planets_m)


#Building test set
test_nh_labels = [1 for x in range(len(test_planets_nh))]
test_p_labels = [2 for x in range(len(test_planets_p))]
test_m_labels = [3 for x in range(len(test_planets_m))]

test_set = pd.concat([test_planets_nh, test_planets_p, test_planets_m])
planet_names = test_set['P. Name'].tolist()
test_set = test_set.drop('P. Name', axis=1)
test_set = test_set.values
test_labels = test_nh_labels + test_p_labels + test_m_labels

for algo, clf in algorithms.items():
    accuracy = 0.0
    total_trials = 0.0
    print('Testing', algo)
    iter_count = 0

    accuracy_1 = [0.0 for x in range(len(test_set))]
    accuracy_2 = [0.0 for x in range(len(test_set))]
    accuracy_3 = [0.0 for x in range(len(test_set))]
    accuracy_o = [0.0 for x in range(len(test_set))]

    for outer_iter in range(TOTAL_OUTER_ITERATIONS):
        for inner_iter in range(TOTAL_INNER_ITERATIONS):

            #Building training set
            train_nh_sub = train_planets_nh.sample(n=1000)
            train_nh_labels = [1 for x in range(len(train_nh_sub))]

            train_p_sub = train_planets_p.sample(n=12)
            train_p_labels = [2 for x in range(len(train_p_sub))]

            train_m_sub = train_planets_m.sample(n=12)
            train_m_labels = [3 for x in range(len(train_m_sub))]

            training_set = pd.concat([train_nh_sub, train_p_sub, train_m_sub])
            training_set = training_set.drop('P. Name', axis=1)
            training_set = training_set.values
            training_labels = train_nh_labels + train_p_labels + train_m_labels

            clf.fit(training_set, training_labels)
            predicted_labels = clf.predict(test_set)

            total_trials += 1
            for i in range(len(predicted_labels)):

                if predicted_labels[i] == test_labels[i]:
                    accuracy_o[i] += 1

                if predicted_labels[i] == 1:
                    accuracy_1[i] += 1

                elif predicted_labels[i] == 2:
                    accuracy_2[i] += 1

                elif predicted_labels[i] == 3:
                    accuracy_3[i] += 1

    #Generates LaTex table
    print('P. Name & Non Habitable & Psychroplanet & Mesoplanet & Overall\\\\')
    print('\hline')
    for i in range(len(predicted_labels)):
        print(planet_names[i], '&', accuracy_1[i]*100/total_trials,
        '&', accuracy_2[i]*100/total_trials, '&', accuracy_3[i]*100/total_trials,
        '&', accuracy_o[i]*100/total_trials, '\\\\')
    print('\hline')
