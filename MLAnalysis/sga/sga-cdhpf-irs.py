import numpy as np
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def sga(rate, alp, bet, gam, delt, UL):
    LL = 0.001
    #UL = 0.1
    count = 0

    while (alp < UL and
            alp > LL and
            bet < UL and
            bet > LL and
            gam < UL and
            gam > LL and
            delt < UL and
            delt > LL and
            alp + bet + gam + delt <UL):

            alp = alp + rate*(np.log(R)*(R**alp)*(D**bet)*(T**gam)*(V**delt))
            bet = bet + rate*(np.log(D)*(R**alp)*(D**bet)*(T**gam)*(V**delt))
            gam = gam + rate*(np.log(T)*(R**alp)*(D**bet)*(T**gam)*(V**delt))
            delt = delt + rate*(np.log(V)*(R**alp)*(D**bet)*(T**gam)*(V**delt))

            if rate - (randint(0,100)/100000000) > 0:
                rate = rate - (randint(0,100)/100000000)

            #print(alph, bet, gam, delt, rate)
            count += 1

    cdhs = (R**alp)*(D**bet)*(T**gam)*(V**delt)
    return(alp, bet, gam, delt, rate, cdhs)

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
cdhs = []

ll = 0
for i in range(0, 19):
    ll = ll + 0.05
    ul = ll + 1.5
    #a, b, g, d, r, c = sga(r, a, b, g, d, ll, ul)
    a, b, g, d, r, c = sga(r, a, b, g, d, ul)
    #a, b, g, d, r, c = sga(r, a, b, g, d)
    alpha.append(a)
    beta.append(b)
    gamma.append(g)
    delta.append(d)
    cdhs.append(c)
    print(a, b, g, d, c)

print(ul)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(alpha, beta, cdhs, c='r')
ax.scatter(alpha, beta, cdhs, c='r', marker = '^')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.set_zlabel('CDHS')

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(beta, gamma, cdhs, c='r')
ax.scatter(beta, gamma, cdhs, c='r', marker = '^')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel('CDHS')
"""
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
