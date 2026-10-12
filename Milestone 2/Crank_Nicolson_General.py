# %%
import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def crank_nicholson(F, U0, At, t):
    def G(X):
        return X-U0-0.5*At*(F(U0, t)+F(X, t+At))

    # Use Newton's method to solve for the next time step
    from scipy.optimize import newton
    return newton(func=G, x0=U0)
