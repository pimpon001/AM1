import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def Runge_Kutta4(F, U, At, t):
    k1 = F(U, t)
    k2 = F(U + 0.5 * At * k1, t + 0.5 * At)
    k3 = F(U + 0.5 * At * k2, t + 0.5 * At)
    k4 = F(U + At * k3, t + At)
    return U + (At / 6) * (k1 + 2 * k2 + 2 * k3 + k4)