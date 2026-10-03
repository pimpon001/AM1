# %%
from numpy import array, zeros
import matplotlib.pyplot as plt

N=1000 ; Dt=0.1
Nv=2 
U=zeros((N+1,Nv))
U[0,:]=array([1, 0])

def f(U):
    return array([U[1],-U[0]])

def Euler(U0, F, N, Dt):
    U = zeros((N+1, len(U0)))
    U[0,:] = U0
    for n in range(0, N):
        U[n+1,:] = U[n,:] + Dt * F(U[n,:])
    return U

U = Euler(U[0,:], f, N, Dt)

plt.plot(U[:,0],U[:,1])
plt.xlabel('U[0]')
plt.ylabel('U[1]') 
plt.title('Cauchy Problem')
plt.grid()
plt.axis('equal')
plt.show()

# %%
