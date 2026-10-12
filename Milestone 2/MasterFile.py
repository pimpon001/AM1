from numpy import array, zeros, sin, cos, pi
import matplotlib.pyplot as plt
from scipy.optimize import newton 

def Oscilador(U,t):
    return array([U[1], -U[0]])

def partition(a,b,N):
    """Devuelve un vector de N+1 puntos equiespaciados entre a y b. N es el número de segmentos"""
    return array([a + i*(b-a)/N for i in range( 0, N+1)])

U = cauchy_problem( F = oscilador, U0 = array([1, 0]), t = partition(a=0, b=10, N=100), scheme = euler)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Euler")
plt.axis('equal')
plt.grid(True)
plt.show()

U = cauchy_problem( F = oscilador, U0 = array([1, 0]), t = partition(a=0, b=10, N=100), scheme = rk4)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Runge-Kutta de 4º Orden")
plt.axis('equal')
plt.grid(True)
plt.show()

U = cauchy_problem( F = oscilador, U0 = array([1, 0]), t = partition(a=0, b=10, N=100), scheme = crank_nicholson)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Crank-Nicolson")
plt.axis('equal')
plt.grid(True)
plt.show()
