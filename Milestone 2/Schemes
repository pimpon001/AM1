from numpy import array, zeros, sin, cos, pi, linspace
import matplotlib.pyplot as plt
from scipy.optimize import newton

"CAUCHY_PROBLEM"
def cauchy_problem(F, U0, t, scheme):
    """
    Solves the Cauchy problem for a system of ODEs using the specified scheme.

    Parameters:
    F : function
        The function defining the system of ODEs.
    U0 : array
        The initial condition.
    t : array
        The time points at which to evaluate the solution.
    scheme : function
        The numerical scheme to use for solving the ODEs.


    Returns:
    U : array
        The solution array containing the values of U at each time step.
    """
    Nv = len(U0)
    U = zeros((len(t), Nv))
    U[0, :] = U0

    for n in range(len(t) - 1):
        U[n + 1, :] = scheme(F, U[n, :], t[n + 1] - t[n], t[n])

    return U

"RUNGE_KUTTA_4"
def Runge_Kutta4(F, U0, At, t):
    k1 = F(U0, t)
    k2 = F(U0 + 0.5 * At * k1, t + 0.5 * At)
    k3 = F(U0 + 0.5 * At * k2, t + 0.5 * At)
    k4 = F(U0 + At * k3, t + At)
    return U0 + (At / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

"INVERSE_EULER"
def inverse_euler(F, U0, At, t):
    def G(X):
        return X-U0-At*F(X, t+At)

    return newton(func=G, x0=U0)

"EULER"
def Euler_Method(F, U0, At, t):
    return U0 + At*F(U0,t)

"CRANK_NICHOLSON"
def crank_nicholson(F, U0, At, t):
    def G(X):
        return X-U0-0.5*At*(F(U0, t)+F(X, t+At))
    
    return newton(func=G, x0=U0)