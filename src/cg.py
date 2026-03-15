import numpy as np
from .utils import apply_M_inv

def cg(A, b, x0=None, M=None, tol=1e-6, max_iter=1000):
    """
    Preconditioned Conjugate Gradient method.
    Suitable for simmetric positive definite (SPD) matrix only.
    """

    N = len(b)
    
    if x0 is None:
        x0 = np.zeros(N)
    
    x = x0.copy()

    r = b - A @ x
    z = apply_M_inv(M, r)
    p = z.copy()

    rho = np.dot(r, z)
    res_history = [np.linalg.norm(r)]

    tol_abs = tol * res_history[0] if res_history[0] != 0 else tol 

    for i in range(max_iter):
        if res_history[-1] < tol_abs:
            break

        q = A @ p 
        alpha = rho / np.dot(p, q)

        x = x + alpha * q
        r = r - alpha * q
        z = apply_M_inv(M, r)

        rho_new = np.dot(r, z)
        beta = rho_new / rho 
        p = z + beta * p 

        rho = rho_new 
        res_history.append(np.linalg.norm(r))

    return x, res_history