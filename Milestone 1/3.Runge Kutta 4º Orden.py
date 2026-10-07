import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

N = 100000
Nv = 2
At = 0.0001

U = zeros((N, Nv))
U[0,:] = array([1, 0])

def F(U):
    return array([U[1], -U[0]]) #Ecuacion del oscilador armonico simple

for n in range(N-1):
    k1 = At * F(U[n,:])
    k2 = At * F(U[n,:] + 0.5 * k1)
    k3 = At * F(U[n,:] + 0.5 * k2)
    k4 = At * F(U[n,:] + k3)
    U[n+1,:] = U[n,:] + (k1 + 2*k2 + 2*k3 + k4) / 6

plt.plot(U[:,0], U[:,1])
plt.xlabel('x')
plt.ylabel('v')
plt.title('Oscilador Armónico Simple - Método de Runge Kutta 4º Orden')
plt.grid()
plt.show()
