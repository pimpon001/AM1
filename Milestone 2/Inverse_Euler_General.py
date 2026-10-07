import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def inverse_euler(F, U0, At, N, tol, itmax):

    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0

    for n in range(N):
        U[n + 1, :] = U[n, :] + At * F(U[n, :])  # Predicción usando Euler

    
    for k in range(itmax):
        U_new = U[n, :] + At * F(U[n + 1, :])  # Actualización usando Inverse Euler
        if abs(U_new - U[n + 1, :]).max() < tol:
            break
        U[n + 1, :] = U_new
       
    return U