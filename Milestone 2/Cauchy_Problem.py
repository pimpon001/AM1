#%%
import matplotlib.pyplot as plt
from numpy import arange, array, zeros, sin, cos, pi
#Numerical Methods
from EulerMethod_General import Euler_Method
#from Inverse_Euler_General import inverse_euler
from Crank_Nicolson_General import Crank_Nicolson
from Runge_Kutta4_General import Runge_Kutta4
#Error Comparison Function
from ErrorFunction import Plot_Error

N=100  # Number of time steps
At=0.1  # Time step size
tol=1e-6  # Tolerance for iterative methods
itmax=100  # Maximum number of iterations for iterative methods
t = arange(N + 1) * At
Uex = array([cos(t), -sin(t)]).T

def F(U):
    # Example function for the Cauchy problem
    return array([U[1], -U[0]])

Nv=2
U0=array([1.0, 0.0])  # Initial condition


def Cauchy_Problem(F, U0, At, N, tol, itmax):
    # Solve the Cauchy problem using different numerical methods
    U_Euler = Euler_Method(F, U0, At, N)
    #U_Inverse_Euler = inverse_euler(F, U0, At, N, tol, itmax)
    U_Crank_Nicolson = Crank_Nicolson(F, U0, At, N, tol, itmax)
    U_Runge_Kutta4 = Runge_Kutta4(F, U0, At, N)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(U_Euler[:, 0], label='Euler Method', linestyle='--')
    #plt.plot(U_Inverse_Euler[:, 0], label='Inverse Euler Method', linestyle='-.')
    plt.plot(U_Crank_Nicolson[:, 0], label='Crank-Nicolson Method', linestyle=':')
    plt.plot(U_Runge_Kutta4[:, 0], label='Runge-Kutta 4 Method', linestyle='-')
    plt.title('Cauchy Problem Solutions')
    plt.xlabel('Time Steps')
    plt.ylabel('Solution Value')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot error comparison
    Plot_Error(At, Uex, [U_Euler, U_Crank_Nicolson, U_Runge_Kutta4], ['Euler', 'Crank-Nicolson', 'Runge-Kutta 4'])
# %%
