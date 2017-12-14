import sys
import csv
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

cdhs = pd.read_csv('catalogs/cdhs.csv')
esi = pd.read_csv('_data_/phl_hec_all_confirmed.csv',
usecols=['P. Name', 'P. ESI'],skipinitialspace=True)
merged_catalogs = pd.merge(cdhs, esi, on='P. Name')
print(merged_catalogs)
crs_c,crs_p = pearsonr(merged_catalogs['P. ESI'], merged_catalogs['CRS_Yscore'])
drs_c,drs_p = pearsonr(merged_catalogs['P. ESI'], merged_catalogs['DRS_Yscore'])
ncrs_c,ncrs_p = pearsonr(merged_catalogs['P. ESI'],
merged_catalogs['CRS_Norm'])
ndrs_c,ndrs_p = pearsonr(merged_catalogs['P. ESI'],
merged_catalogs['DRS_Norm'])

print('Correlation between CDHS-crs and ESI: ', crs_c)
print('p-Value: ', crs_p)
print('-------------------')
print('Correlation between CDHS-drs and ESI: ', drs_c)
print('p-Value: ', drs_p)
print('-------------------')
print('Correlation between CDHS-crs (normalaized) and ESI: ', ncrs_c)
print('p-Value: ', ncrs_p)
print('-------------------')
print('Correlation between CDHS-drs and ESI: ', ndrs_c)
print('p-Value: ', ndrs_p)
print('-------------------')

#LaTex output
print('Scores & Correlation\\\\')
print('\hline')
print('CDHS-crs and ESI &', crs_c, '\\\\')
print('CDHS-drs and ESI &', drs_c, '\\\\')
print('CDHS-crs (normalaized) and ESI &', ncrs_c, '\\\\')
print('CDHS-drs (normalaized) and ESI &', ndrs_c, '\\\\')
print('-------------------')
