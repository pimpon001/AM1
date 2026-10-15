import numpy as np

def My_Newton(U:array, At:float, t:float, F:Callable[[array,float], array],
              tol:float=1e-10, max_iter:int=50):
    # Método de Newton para resolver ecuaciones no lineales.
    Y = U + At * F(U, t)
    for _ in range(max_iter):
        R = Y - U - At * F(Y, t + At)
        if norm(R) <= tol:
            return Y
        J = np.eye(len(U)) - At * My_Jacobian(F, Y, t + At)
        Y = Y - np.linalg.solve(J, R)

    raise RuntimeError(f"Newton no convergió en {max_iter} iteraciones.")