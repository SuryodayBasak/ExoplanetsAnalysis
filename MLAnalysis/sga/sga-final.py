import numpy as np
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def sga(rate, a1, a2, F1, F2, UL):
    LL = 0.001
    #UL = 0.1
    count = 0

    while (a1 < UL and
            a1 > LL and
            a2 < UL and
            a2 > LL and
            a1 + a2 <UL):

            a1 = a1 + rate*(np.log(F1)*(F1**a1)*(F2**a2))
            a2 = a2 + rate*(np.log(F2)*(F1**a1)*(F2**a2))

            #print(rate)
            dec = (randint(0,100)/100000000)
            if rate -  dec > 0:
                rate = rate - (randint(0,10000)/100000000)
            #print(alph, bet, gam, delt, rate)
            count += 1

    cdhsx = (F1**a1)*(F2**a2)
    return a1, a2, rate, cdhsx

R = 1.09
D = 1
T = 396.5/288
#T = 1833.5
V = 2.02

a = 0.01
b = 0.01
g = 0.01
d = 0.01
r = 0.005

alpha = []
beta = []
gamma = []
delta = []

ll = 0
ll = ll + 0.05
ul = ll + 1
a, b, r, c1 = sga(r, a, b, R, D, ul)
print(a, b, c1)

print('---')
r = 0.005
ll = 0

ll = ll + 0.05
ul = ll + 1
g, d, r, c2 = sga(r, g, d, R, D, ul)
print(g, d, c2)

print("CDHS is:\t", (0.5*c1) + (0.5*c2))

plt.show()
