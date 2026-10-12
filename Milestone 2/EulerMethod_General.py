# %%
import matplotlib.pyplot as plt
from numpy import array, zeros, sin, cos, pi

def Euler_Method(F, U, At, t):
    return U + At*F(U,t)