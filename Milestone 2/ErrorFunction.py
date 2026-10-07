#%%
import matplotlib.pyplot as plt
from numpy import arange
from numpy.linalg import norm

def Plot_Error(At, Uex, Us, labels):
    # Dt: paso de tiempo
    # Uex: solución exacta, array (N+1, Nv)
    # Us: lista de soluciones numéricas, cada una (N+1, Nv)
    # labels: lista de nombres, uno por solución
    N = Uex.shape[0] - 1
    t = arange(N + 1) * At

    #if __name__ == '__main__':
    plt.figure()
    for U, lab in zip(Us, labels):
        err = norm(U - Uex, axis=1)            # error en cada instante
        plt.semilogy(t[1:], err[1:], label=lab)  # escala log; se omite t=0

    plt.xlabel('t')
    plt.ylabel('||U - U_exacta||')
    plt.title('Error frente al tiempo')
    plt.grid(which='both', alpha=0.4)
    plt.legend()

# %%
