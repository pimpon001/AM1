from numpy import array, zeros, sin, cos, pi
import matplotlib.pyplot as plt
from scipy.optimize import newton
import Schemes
from Schemes import cauchy_problem, Runge_Kutta4, inverse_euler, Euler_Method, crank_nicholson
from MasterFile import partition

"Estructura del código:"
"Se le pide al usuario que elija entre varias funciones y que elija una condición inicial "
"Se le pide también que defina el intervalo de tiempo y el tamaño del paso -> Define N "

