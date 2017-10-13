import pandas as pd
import retrieveHECData 
import numpy as np
import matplotlib.pyplot as plt
import random
plt.style.use('ggplot')
fig, axs = plt.subplots(2, 2, sharex=True)
"""
ax = plt.gca()
ax.cla()
ax.set_aspect('equal')
"""
data_object = retrieveHECData.HECDataFrame(download_new_flag = 0)
data_object.populatePreprocessedData()
data_non_hab, data_psychro, data_meso = data_object.returnAllSamples()

"""
Thermal Classification Scheme:

-100 to -50: hP
-50 to 0: P
0 to 50: M
50 to 100: T
100 to 150: hT

label the planet classes in colours
plot vertical lines for class boundaries
"""

#print(data_non_hab['P. Ts Mean (K)'])
#bins = np.linspace(0, 500, 100)
bins = np.linspace(200, 340, 40)
axs[0].hist(data_non_hab['P. Ts Mean (K)'], bins, alpha=0.5, label='NH')
axs[0].hist(data_psychro['P. Ts Mean (K)'], bins, alpha=0.5, label='P')
axs[0].hist(data_meso['P. Ts Mean (K)'], bins, alpha=0.5, label='M')
axs[0].legend(loc='upper right')
axs[0].xlabel("Mean Surface Temperature (K)")
axs[0].ylabel("Frequency of Occurrence")
#plt.grid()
#plt.show()

"""
ax.set_xlim((-60, 100))
plt.plot((-50, -50), (-10, 10), 'b-')
plt.plot((0, 0), (-10, 10), 'g-')
plt.plot((50, 50), (-10, 10), 'r-')
"""
#plt.show() 

"""
Mass Classification Scheme:

0 to 0.00001: Asteroidan
0.00001 to 0.1: Mercurian
0.1 to 0.5: Subterran
0.5 to 2: Terran
2 to 10: Superterran
10 to 50: Neptunian
50 to 5000: Jovian

label the planet classes in colours
plot vertical lines for class boundaries
"""

#print(data_non_hab['P. Ts Mean (K)'])~~~
#bins = np.linspace(0, 500, 100)~~~
bins = np.linspace(0, 28, 15)
#plt.hist(data_non_hab['P. Mass (EU)'], bins, alpha=0.5, label='NH')~~~
axs[1].hist(data_psychro['P. Mass (EU)'], bins, alpha=0.5, label='Psychroplanet')
axs[1].hist(data_meso['P. Mass (EU)'], bins, alpha=0.5, label='Mesoplanet')
axs[1].legend(loc='upper right')
axs[1].xlabel("Mass of Planet")
axs[1].ylabel("Frequency of Occurrence")
#plt.grid()~~~
#plt.show()

"""
Zone Class: Relate this to P. Habitable, P. Temp Mean

Hot -> probably between the parent star and CHZ
Warm -> probably within the CHZ
Cold -> probably beyond the CHZ

label the planet classes in colours
~~~~~plot vertical lines for class boundaries
create the plot with P. Temp Mean as the X axis
Just plot the points on X axis
"""

#bins = np.linspace(0, 1, 3)
#rows = np.random.choice(data_non_hab.index.values, 25)
#nh_subsample = data_non_hab.ix[rows]
#plt.hist(nh_subsample['P. Habitable'], bins, alpha=0.5, label='NH')
#plt.hist(data_psychro['P. Habitable'], bins, alpha=0.5, label='Psychroplanet')
#plt.hist(data_meso['P. Habitable'], bins, alpha=0.5, label='Mesoplanet')
#plt.legend(loc='upper right')
#plt.xlabel("Mass of Planet")
#plt.ylabel("Frequency of Occurrence")
#plt.show()

"""
nh_subsample = data_non_hab.ix[rows]
zone1_nh = nh_subsample[nh_subsample['P. Zone Class'] == 1.0]
zone2_nh = nh_subsample[nh_subsample['P. Zone Class'] == 2.0]
zone3_nh = nh_subsample[nh_subsample['P. Zone Class'] == 3.0]
"""

"""
zone1_nh = data_non_hab[data_non_hab['P. Zone Class'] == 1.0]
zone2_nh = data_non_hab[data_non_hab['P. Zone Class'] == 2.0]
zone3_nh = data_non_hab[data_non_hab['P. Zone Class'] == 3.0]

zone1_p = data_psychro[data_psychro['P. Zone Class'] == 1.0]
zone2_p = data_psychro[data_psychro['P. Zone Class'] == 2.0]
zone3_p = data_psychro[data_psychro['P. Zone Class'] == 3.0]

zone1_m = data_meso[data_meso['P. Zone Class'] == 1.0]
zone2_m = data_meso[data_meso['P. Zone Class'] == 2.0]
zone3_m = data_meso[data_meso['P. Zone Class'] == 3.0]
"""
"""
zone1_nh
zone2_nh
zone2_p
zone2_m
zone3_nh
"""

#print(zone1['P. Ts Mean (K)'])
#zone1_temp = zone1['P. Ts Mean (K)']
#print(zone1_temp)
#plt.plot(zone1_nh['P. Ts Mean (K)'],[0 for i in range(len(zone1_nh))],marker='o',linestyle='',color='r')
"""
plt.plot(zone2_nh['P. Ts Mean (K)'],[0 for i in range(len(zone2_nh))],marker='o',linestyle='',color='b')
plt.plot(zone2_p['P. Ts Mean (K)'],[0 for i in range(len(zone2_p))],marker='o',linestyle='',color='g')
plt.plot(zone2_m['P. Ts Mean (K)'],[0 for i in range(len(zone2_m))],marker='o',linestyle='',color='r')
"""
#plt.plot(zone3_nh['P. Ts Mean (K)'],[0 for i in range(len(zone3_nh))],marker='o',linestyle='',color='g')

#print(zone3)
#print(zone2)
#print(zone3_nh)
#print(zone2_p)
#print(zone2_m)
"""
plt.legend(loc='upper right')
plt.xlabel("Surface Temperature")
#plt.ylabel("Frequency of Occurrence")
plt.show()
"""

"""
Composition Class: S. Fe/H?

Iron:1.0
rocky-iron:2.0
rocky-water:3.0
"""

#print(data_non_hab['P. Ts Mean (K)'])~~~
#bins = np.linspace(0, 500, 100)~~~
bins = np.linspace(-1, 0.5, 20)
#plt.hist(data_non_hab['S. [Fe/H]'], bins, alpha=0.5, label='NH')~~~
axs[2].hist(data_psychro['S. [Fe/H]'], bins, alpha=0.5, label='Psychroplanet')
axs[2].hist(data_meso['S. [Fe/H]'], bins, alpha=0.5, label='Mesoplanet')
axs[2].legend(loc='upper right')
axs[2].xlabel("S. Fe/H")
axs[2].ylabel("Frequency of Occurrence")
#plt.grid()~~~
#plt.show()

"""
Atmosphere Classification: Relate with HZA

none:1.0
metals-rich:2.0
hydrogen-rich:3.0
"""
#print(data_non_hab['P. Ts Mean (K)'])~~~
bins = np.linspace(-1, 2.5, 20)
#plt.hist(data_non_hab['P. HZA'], bins, alpha=0.5, label='NH')
axs[4].hist(data_psychro['P. HZA'], bins, alpha=0.5, label='Psychroplanet')
axs[4].hist(data_meso['P. HZA'], bins, alpha=0.5, label='Mesoplanet')
axs[4].legend(loc='upper right')
axs[4].xlabel("P. HZA")
axs[4].ylabel("Frequency of Occurrence")
#plt.grid()~~~
plt.show()
