import numpy as np

alpha = 0.01
beta = 0.01
gamma = 0.01
delta = 0.01
rate = 0.001

LL = 0.0
UL = 1.0
count = 0


R = 1.99
D = 1.03
T = 1833.5/288
V = 2.02

while (alpha < UL and
        alpha > LL and
        beta < UL and
        beta > LL and
        gamma < UL and
        gamma > LL and
        delta < UL and
        delta > LL and
        alpha + beta + gamma + delta <UL):

    alpha = alpha + rate*(np.log(R)*(R**alpha)*(D**beta)*(T**gamma)*(V**delta))
    beta = beta + rate*(np.log(D)*(R**alpha)*(D**beta)*(T**gamma)*(V**delta))
    gamma = gamma + rate*(np.log(T)*(R**alpha)*(D**beta)*(T**gamma)*(V**delta))
    delta = delta + rate*(np.log(V)*(R**alpha)*(D**beta)*(T**gamma)*(V**delta))
    print(alpha, beta, gamma, delta)

CDHS = (R**alpha)*(D**beta)*(T**gamma)*(V**delta)
print('CDHS Score:\t', CDHS)
