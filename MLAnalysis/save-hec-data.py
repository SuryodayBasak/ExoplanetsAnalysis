import sys
import csv
import numpy as np
import pandas as pd
import retrieveHECData

np.set_printoptions(precision=2)

#Retrieving data from PHL-HEC
data_object = retrieveHECData.HECDataFrame(download_new_flag = 0)
data_object.populatePreprocessedData()
data_nh, data_p, data_m = data_object.returnAllSamples()

lbl_nh = [0 for x in range(len(data_nh))]
lbl_p = [1 for x in range(len(data_p))]
lbl_m = [2 for x in range(len(data_m))]

all_classes = [data_nh, data_p, data_m]
data = pd.concat(all_classes, ignore_index=True)
lbls = lbl_nh + lbl_p + lbl_m
lbls = pd.DataFrame({'hab_lbl':lbls})

#print(np.shape(dataset))
#print(len(lbls))
dataset = pd.concat([lbls, data], axis=1)
dataset.to_csv('dataset.csv')
