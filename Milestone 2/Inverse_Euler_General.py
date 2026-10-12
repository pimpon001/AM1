import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi
from scipy.optimize import newton

def inverse_euler(F, U, At, t):
    def G(X):
        return X-U-At*F(X, t+At)

    return newton(func=G, x0=U)