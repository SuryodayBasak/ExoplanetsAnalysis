import requests
import sys
import zipfile
import pandas as pd
from io import StringIO
import os
import shutil
import listOfFeatures

#class HECDataFrame:
os.mkdir('_temp_')
url = 'http://www.hpcf.upr.edu/~abel/phl/phl_hec_all_confirmed.csv.zip'  
print("Attempting to download dataset.")
r = requests.get(url)

try:
	with open("_temp_/source.zip", "wb") as code:
		code.write(r.content)
	archive = zipfile.ZipFile('_temp_/source.zip', 'r')
	archive.extractall('_temp_')
	print("Data retrieval successful.")

except:
	print("Error in downloading file!")
	shutil.rmtree('_temp_')
	sys.exit(0)
	
featuresList = listOfFeatures.HECFeatures()
print('Attempting to extract the data as required for ML analysis.')
try:
	data = pd.read_csv('_temp_/phl_hec_all_confirmed.csv', usecols=featuresList.returnFeatureNames())
	print('Data extraction successful.')
except:
	print('An error occured while reading the required features from the catalog.')
	shutil.rmtree('_temp_')
shutil.rmtree('_temp_')
