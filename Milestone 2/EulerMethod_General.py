# %%
import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def Crank_Nicolson(F, U0, At, N):

    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0

    for n in range(N):
        U[n + 1, :] = U[n, :] + At * F(U[n, :])

    return U    
