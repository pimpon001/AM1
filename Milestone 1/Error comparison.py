#%%
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numpy import array, arange, cos, sin, zeros

N = 10000
# Paso de integración (tiempo discreto)
At = 0.0001
# Número de variables del estado
Nv = 2

t = arange(N + 1) * At
Uex = array([cos(t), -sin(t)]).T

def F(U):
    # Derivada del estado U = [x, v]
    return array([U[1], -U[0]])  # Usamos sin(x) para un oscilador no lineal, o -x para lineal

UCN = zeros((N + 1, Nv)) 
UCN[0, :] = array([1, 0])  

E= UCN.copy() #Copia de la matriz para almacenar los resultados del método de Euler
E[0, :] = array([1, 0])

tol=1e-6  
itmax=100  

for n in range(0, N):
    E[n + 1, :] = E[n, :] + At * F(E[n, :])  # Método de Euler explícito

for n in range(0, N):
    Fn = F(UCN[n, :])  
    UEuler= UCN[n, :] + At * Fn  

    for k in range(itmax):
        Fk = F(UEuler)  # Evaluamos la función F en la predicción
        UEuler_new = UCN[n, :] + (At / 2) * (Fn + Fk)  # Actualizamos la predicción usando Crank-Nicolson
        if abs(UEuler_new - UEuler).max() < tol:  # Verificamos la convergencia
            break
        UEuler = UEuler_new  # Actualizamos UEuler para la siguiente iteración

    UCN[n + 1, :] = UEuler_new


def Plot_Error(At, Uex, Us, labels):
    # Dt: paso de tiempo
    # Uex: solución exacta, array (N+1, Nv)
    # Us: lista de soluciones numéricas, cada una (N+1, Nv)
    # labels: lista de nombres, uno por solución
    N = Uex.shape[0] - 1
    t = arange(N + 1) * At

    plt.figure()
    for U, lab in zip(Us, labels):
        err = norm(U - Uex, axis=1)            # error en cada instante
        plt.semilogy(t[1:], err[1:], label=lab)  # escala log; se omite t=0

    plt.xlabel('t')
    plt.ylabel('||U - U_exacta||')
    plt.title('Error frente al tiempo')
    plt.grid(which='both', alpha=0.4)
    plt.legend()

def Plot_Diferencia(At, U1, U2, label1='Euler', label2='Crank-Nicolson'):
    # At: paso de tiempo
    # U1, U2: dos soluciones numéricas, arrays (N+1, Nv) del mismo tamaño
    N = U1.shape[0] - 1
    t = arange(N + 1) * At

    plt.figure()
    dif = norm(U1 - U2, axis=1)               # distancia entre ambas en cada instante
    plt.semilogy(t[1:], dif[1:])              # escala log; se omite t=0 (dif = 0)

    plt.xlabel('t')
    plt.ylabel(f'||{label1} - {label2}||')
    plt.title(f'Diferencia entre {label1} y {label2}')
    plt.grid(which='both', alpha=0.4)

Plot_Error(At, Uex, [E, UCN], ['Euler', 'Crank-Nicolson'])
Plot_Diferencia(At, E, UCN)
plt.show()
# %%
