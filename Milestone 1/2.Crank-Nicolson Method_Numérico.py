# %%
import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

# Modelo físico: oscilador armónico simple sin amortiguamiento.
# La variable U = [x, v] representa posición x y velocidad v.
# La ecuación del sistema es:
#   x' = v
#   v' = -x
# que corresponde a x'' + x = 0.
# Esto describe un movimiento circular/armónico en el plano (x, v).
# Número de pasos de tiempo
N = 100000
# Paso de integración (tiempo discreto)
At = 1
# Número de variables del estado
Nv = 2

UCN = zeros((N + 1, Nv))  # Matriz para almacenar los resultados del método de Crank-Nicolson
UCN[0, :] = array([1, 0])  # Condición inicial

def F(U):
    # Derivada del estado U = [x, v]
    return array([U[1], -sin(U[0])])  # Usamos sin(x) para un oscilador no lineal, o -x para lineal

tol=1e-6  # Tolerancia para la convergencia del método iterativo
itmax=100  # Número máximo de iteraciones para que no se quede atrapado en un bucle infinito

#Usamos Euler como punto de entrada para el método de Crank-Nicolson, que es implícito y requiere resolver un sistema de ecuaciones en cada paso de tiempo.
for n in range(0, N):
    Fn = F(UCN[n, :])  # Evaluamos la función F en el estado actual
    E= UCN[n, :] + At * Fn  # Predicción usando Euler

    for k in range(itmax):
        Fk = F(E)  # Evaluamos la función F en la predicción
        E_new = UCN[n, :] + (At / 2) * (Fn + Fk)  # Actualizamos la predicción usando Crank-Nicolson
        if abs(E_new - E).max() < tol:  # Verificamos la convergencia
            break
        E = E_new  # Actualizamos E para la siguiente iteración

    UCN[n + 1, :] = E_new  # Guardamos el estado actualizado en la matriz de resultados
        

# Graficamos la trayectoria en el plano (x, v), que para este sistema
# es una órbita circular alrededor del origen.
plt.plot(UCN[:, 0], UCN[:, 1])
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico simple')
plt.show()
# %%
