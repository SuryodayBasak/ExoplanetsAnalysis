import numpy as np

alpha = 0.01
beta = 0.01
gamma = 0.01
delta = 0.01
rate = 0.001

LL = 0.01
UL = 1.0
count = 0

R = 19.04
D = 0.64
T = 11.1
V = 34.2

while (alpha < UL and
        beta < UL and
        gamma < UL and
        delta < UL and
        alpha + beta + gamma + delta <UL):
    alpha = alpha + rate*(1/R)
    beta = beta + rate*(1/D)
    gamma = gamma + rate*(1/T)
    delta = delta + rate*(1/V)
    print(alpha, beta, gamma, delta)

CDHS = (R**alpha)*(D**beta)*(T**gamma)*(V**delta)
print('CDHS Score:\t', CDHS)
