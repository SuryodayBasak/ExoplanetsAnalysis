import numpy as np
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

R = 5
D = 5
T = 5
#T = 396.5/288
#T = 1833.5
V = 5

a = 0.01
b = 0.01
g = 0.01
d = 0.01
r1 = 0.005
r2 = 0.005
c1 = 0
c2 = 0

alpha = []
beta = []
gamma = []
delta = []
cdhs_i = []
cdhs_e = []
cdhs = []

fig = plt.figure()
plt.title('Gradient ascent towards maxima of the CDHS score of 55 Cnc e')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}
plt.rc('font', **font)

ax1 = fig.add_subplot(121, projection = '3d')
ax1.set_xlabel(r'$\alpha$' + '  (alpha)')
ax1.set_ylabel(r'$\beta$' + '  (beta)')
ax1.set_zlabel('CDHS_i')
ax1.set_xlim3d(0, 1)
ax1.set_ylim3d(0, 1)
ax1.set_zlim3d(0, 5)

ax2 = fig.add_subplot(122, projection = '3d')
ax2.set_xlabel(r'$\gamma$' + '  (gamma)')
ax2.set_ylabel(r'$\delta$' + '  (delta)')
ax2.set_zlabel('CDHS_e')
ax2.set_xlim3d(0, 1)
ax2.set_ylim3d(0, 1)
ax2.set_zlim3d(0, 5)
#fig.subplots_adjust(left=-0.5, right=-0.9, top=-0.9, bottom=-0.5)

ll = 0
img_index = 0
for i in range(0, 19):
    ll = ll + 0.05
    ul = ll + 0.05
    a1, b1, r11, c11 = sga(r1, a, b, R, D, ul)
    g1, d1, r21, c21 = sga(r2, g, d, R, D, ul)

    alpha.append(a)
    beta.append(b)
    gamma.append(g)
    delta.append(d)

    cdhs_i.append(c1)
    cdhs_e.append(c2)
    cdhs.append((0.5*c11)*(0.5*c21))
    print(a, b, g, d, c1, c2)

    ax1.plot([a, a1], [b, b1], [c1, c11], c='r')
    ax2.plot([g, g1], [d, d1], [c2, c21], c='r')
    ax1.scatter(a1, b1, c11, c='r', marker = '^')
    ax2.scatter(g1, d1, c21, c='r', marker = '^')

    a = a1
    b = b1
    g = g1
    d = d1
    c1 = c11
    c2 = c21
    r1 = r11
    r2 = r21

    plt.savefig('movie-frames/' + str(img_index)+'.png')
    img_index = img_index + 1
print('---')

plt.show()
