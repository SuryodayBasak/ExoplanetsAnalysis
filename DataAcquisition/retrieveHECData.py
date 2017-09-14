"""
This is a module which can be used to automate the process of downloading
and preprocessing the data of PHL-HEC. This will make the latest catalog
available for ML analysis.

The dataframes returned will be those of psychroplanets, mesoplanets, and
non-habitable planets, of iron, rocky, and rocky-iron planets.
"""

import requests
import sys
import zipfile
import pandas as pd
import os
import shutil
import numpy as np

class HECFeatures:
	def __init__(self):
		self.feature_names = []
		print('Collecting list of features.')
		with open('featuresUsed.txt') as fp:
			for line in fp:
				self.feature_names.append(line[:-1])
	
	def returnFeatureNames(self):
		return self.feature_names
		
class HECDataFrame:
	def __init__(self):
		
		self.zoneClassDict = {'Hot':1.0, 'Warm':2.0, 'Cold':3.0}
		self.massClassDict = {'Mercurian':1.0, 'Subterran':2.0, 'Terran':3.0, 'Superterran':4.0, 'Neptunian':5.0, 'Jovian':6.0}
		self.compClassDict = {'iron':1.0, 'rocky-iron':2.0, 'rocky-water':3.0}
		self.atmoClassDict = {'none':1.0, 'metals-rich':2.0, 'hydrogen-rich':3.0}
		
		dirs_list = os.listdir()
		if '_exo_temp_' in dirs_list:
			print('Existing _exo_temp_ folder found. Removing it.')
			shutil.rmtree('_exo_temp_')
			
		print('Initializing data acquiring object.')
		os.mkdir('_exo_temp_')
		url = 'http://www.hpcf.upr.edu/~abel/phl/phl_hec_all_confirmed.csv.zip'  
		print("Attempting to download dataset.")
		r = requests.get(url)

		try:
			with open("_exo_temp_/source.zip", "wb") as code:
				code.write(r.content)
			archive = zipfile.ZipFile('_exo_temp_/source.zip', 'r')
			archive.extractall('_exo_temp_')
			print("Data retrieval successful.")

		except:
			print("Error in downloading file!")
			shutil.rmtree('_exo_temp_')
			sys.exit(0)
	
		featuresList = HECFeatures()
		print('Attempting to extract the data as required for ML analysis.')
		try:
			self.data = pd.read_csv('_exo_temp_/phl_hec_all_confirmed.csv', usecols=featuresList.returnFeatureNames())
			print('Data extraction successful.')
		except:
			print('An error occured while reading the required features from the catalog.')
			shutil.rmtree('_exo_temp_')
		shutil.rmtree('_exo_temp_')
		
	def extractSamplesFromEachClass(self):
		self.rockyPlanetsDataFrame = self.data[self.data['P. Composition Class'].isin(['iron', 'rocky-iron', 'rocky-water'])]
		
		self.rockyPlanetsDataFrame['P. Zone Class'] = self.rockyPlanetsDataFrame['P. Zone Class'].map(self.zoneClassDict)
		self.rockyPlanetsDataFrame['P. Mass Class'] = self.rockyPlanetsDataFrame['P. Mass Class'].map(self.massClassDict)
		self.rockyPlanetsDataFrame['P. Composition Class'] = self.rockyPlanetsDataFrame['P. Composition Class'].map(self.compClassDict)
		self.rockyPlanetsDataFrame['P. Atmosphere Class'] = self.rockyPlanetsDataFrame['P. Atmosphere Class'].map(self.atmoClassDict)
		
		self.mesoplanetSamples_raw = self.rockyPlanetsDataFrame[self.rockyPlanetsDataFrame['P. Habitable Class'] == 'mesoplanet']
		self.psychroplanetSamples_raw = self.rockyPlanetsDataFrame[self.rockyPlanetsDataFrame['P. Habitable Class'] == 'psychroplanet']
		self.nonhabitableSamples_raw = self.rockyPlanetsDataFrame[self.rockyPlanetsDataFrame['P. Habitable Class'] == 'non-habitable']
		
	def preprocessData(self, classDataFrame):
		_,missing_feature_indexes = np.where(pd.isnull(classDataFrame))
		missing_feature_indexes = np.unique(missing_feature_indexes)
		
		for column in missing_feature_indexes:
			#print(column)
			feature_values = np.array(classDataFrame.ix[:,column])
			feature_values_not_null = feature_values[np.logical_not(np.isnan(feature_values))]
			feature_mean_value = np.mean(feature_values_not_null)
			
			nan_indexes = np.where(np.isnan(feature_values))
			for i in range(len(feature_values)):
				if np.isnan(feature_values[i]):
					feature_values[i] = feature_mean_value
					
			classDataFrame.ix[:,column] = feature_values
		#print(classDataFrame.shape)
		classDataFrame = classDataFrame.drop('P. Habitable Class', axis=1)
		return classDataFrame
		#print(classDataFrame)
			
	def returnPreprocessedData(self):
		self.extractSamplesFromEachClass()
		
		self.nonhabitableSamples_preprocessed = pd.DataFrame(self.preprocessData(self.nonhabitableSamples_raw))
		self.psychroplanetSamples_preprocessed = pd.DataFrame(self.preprocessData(self.psychroplanetSamples_raw))
		self.mesoplanetSamples_preprocessed = pd.DataFrame(self.preprocessData(self.mesoplanetSamples_raw))

		return self.nonhabitableSamples_preprocessed, self.psychroplanetSamples_preprocessed, self.mesoplanetSamples_preprocessed
		
testObj = HECDataFrame()
nh, p, m = testObj.returnPreprocessedData()
print(m)
