from numpy import array, zeros, sin, cos, pi, linspace
import matplotlib.pyplot as plt

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