# %%
import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def Crank_Nicolson(F, U0, At, N, tol, itmax):
    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0

    for n in range(N):
        Fn = F(U[n, :])
        E = U[n, :] + At * Fn  # Predicción usando Euler

        for k in range(itmax):
            Fk = F(E)
            E_new = U[n, :] + (At / 2) * (Fn + Fk)  # Actualización usando Crank-Nicolson
            if abs(E_new - E).max() < tol:
                break
            E = E_new

        U[n + 1, :] = E_new

    return U

