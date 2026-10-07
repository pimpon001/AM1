import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def Runge_Kutta4(F, U0, At, N):

    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0

    for n in range(N):
        k1 = F(U[n, :])
        k2 = F(U[n, :] + 0.5 * At * k1)
        k3 = F(U[n, :] + 0.5 * At * k2)
        k4 = F(U[n, :] + At * k3)
        U[n + 1, :] = U[n, :] + (At / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return U