# %%
import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def Euler_Method(F, U0, At, t):
    return U0 + At*F(U0,t)