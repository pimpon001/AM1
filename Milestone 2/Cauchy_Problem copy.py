#%%
import matplotlib.pyplot as plt
from numpy import arange, array, sin, cos
from EulerMethod_General import Euler_Method
from Crank_Nicolson_General import Crank_Nicolson
from Runge_Kutta4_General import Runge_Kutta4
from ErrorFunction import Plot_Error

N = 100
At = 0.1
tol = 1e-6
itmax = 100
t = arange(N + 1) * At
Uex = array([cos(t), -sin(t)]).T

def F(U):
    return array([U[1], -U[0]])

U0 = array([1.0, 0.0])

def Cauchy_Problem(F, U0, At, N, tol, itmax):
    U_Euler = Euler_Method(F, U0, At, N)
    U_CN = Crank_Nicolson(F, U0, At, N, tol, itmax)
    U_RK4 = Runge_Kutta4(F, U0, At, N)
    return U_Euler, U_CN, U_RK4          # <-- return, don't plot here

# Call the function and keep the results
U_Euler, U_CN, U_RK4 = Cauchy_Problem(F, U0, At, N, tol, itmax)

solutions = [U_Euler, U_CN, U_RK4]
labels = ['Euler', 'Crank-Nicolson', 'Runge-Kutta 4']

# 1) Orbits in the (x, v) plane
plt.figure(figsize=(6, 6))
plt.plot(Uex[:, 0], Uex[:, 1], 'k', linewidth=2, label='Exact')
for U, lab in zip(solutions, labels):
    plt.plot(U[:, 0], U[:, 1], label=lab)
plt.axis('equal')
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.title('Orbits')
plt.grid()
plt.legend()

# 2) Position vs time
plt.figure(figsize=(10, 5))
plt.plot(t, Uex[:, 0], 'k', linewidth=2, label='Exact')
for U, lab in zip(solutions, labels):
    plt.plot(t, U[:, 0], label=lab)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution x(t)')
plt.grid()
plt.legend()

# 3) Error vs time
Plot_Error(At, Uex, solutions, labels)

plt.show()      # one show at the end displays all three figures