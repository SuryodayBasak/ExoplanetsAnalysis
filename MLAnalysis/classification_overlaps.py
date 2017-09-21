import pandas as pd
import retrieveHECData 
import numpy as np

data_object = retrieveHECData.HECDataFrame()
data_object.populatePreprocessedData()

"""
Thermal Classification Scheme:

-100 to -50: hP
-50 to 0: P
0 to 50: M
50 to 100: T
100 to 150: hT
"""

"""
Mass Classification Scheme:

0 to 0.00001: Asteroidan
0.00001 to 0.1: Mercurian
0.1 to 0.5: Subterran
0.5 to 2: Terran
2 to 10: Superterran
10 to 50: Neptunian
50 to 5000: Jovian
"""

"""
Zone Class: Relate this to P. Habitable

Hot -> probably between the parent star and CHZ
Warm -> probably within the CHZ
Cold -> probably beyond the CHZ
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
