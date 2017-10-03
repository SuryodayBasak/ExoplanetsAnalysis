import pandas as pd
import retrieveHECData 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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

print(data_non_hab['P. Ts Mean (K)'])
#bins = np.linspace(0, 500, 100)
bins = np.linspace(200, 340, 40)
plt.hist(data_non_hab['P. Ts Mean (K)'], bins, alpha=0.5, label='NH')
plt.hist(data_psychro['P. Ts Mean (K)'], bins, alpha=0.5, label='P')
plt.hist(data_meso['P. Ts Mean (K)'], bins, alpha=0.5, label='M')
plt.legend(loc='upper right')
plt.xlabel("Mean Surface Temperature (K)")
plt.ylabel("Frequency of Occurrence")
#plt.grid()
plt.show()

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

"""
Composition Class: S. Fe/H?

Iron:1.0
rocky-iron:2.0
rocky-water:3.0
"""

"""
Atmosphere Classification: Relate with HZA

none:1.0
metals-rich:2.0
hydrogen-rich:3.0
"""
