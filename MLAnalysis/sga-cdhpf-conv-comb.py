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

            dec = (randint(0,100)/100000000)
            if rate -  dec > 0:
                rate = rate - (randint(0,100)/100000000)
            #print(alph, bet, gam, delt, rate)
            count += 1

    cdhsx = (F1**a1)*(F2**a2)
    return a1, a2, rate, cdhsx

R = 1.99
D = 1.03
T = 1833.5/288
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
cdhs_i = []
cdhs_e = []

ll = 0
for i in range(0, 19):
    ll = ll + 0.05
    ul = ll + 7
    a, b, r, c = sga(r, a, b, R, D, ul)
    alpha.append(a)
    beta.append(b)
    cdhs_i.append(c)
    print(a, b, c)

print('---')
r = 0.005
ll = 0
for i in range(0, 19):
    ll = ll + 0.05
    ul = ll + 7
    g, d, r, c = sga(r, g, d, R, D, ul)
    gamma.append(g)
    delta.append(d)
    cdhs_e.append(c)
    print(g, d, c)

cdhs = []
for i in range(len(cdhs_i)):
    cdhs.append((0.5*cdhs_e[i])*(0.5*cdhs_i[i]))


fig = plt.figure()
ax = fig.add_subplot(121, projection = '3d')
ax.plot(alpha, beta, cdhs_i, c='r')
ax.scatter(alpha, beta, cdhs_i, c='r', marker = '^')
ax.set_xlabel(r'$\alpha$' + '  (alpha)')
ax.set_ylabel(r'$\beta$' + '  (beta)')
ax.set_zlabel('CDHS_i')

#fig = plt.figure()
ax = fig.add_subplot(122, projection = '3d')
ax.plot(gamma, delta, cdhs_e, c='r')
ax.scatter(gamma, delta, cdhs_e, c='r', marker = '^')
ax.set_xlabel(r'$\gamma$' + '  (gamma)')
ax.set_ylabel(r'$\delta$' + '  (delta)')
ax.set_zlabel('CDHS_e')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)
#plt.title('Gradient ascent towards maxima of the CDHS score of 55 Cnc e')
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(gamma, delta, cdhs, c='r')
ax.scatter(gamma, delta, cdhs, c='r', marker = '^')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\delta$')
ax.set_zlabel('CDHS')
"""
plt.show()
