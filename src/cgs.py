import numpy as np

def cgs(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Conjugate Gradient Squared method.
    Suitable for general (asymmetric) matrices. Transpose-free.
    """

    N = len(b)

    if x0 is None:
        x0 = np.zeros(N)
    
    x = x0.copy()

    r = b - A @ x
    r0_star = r.copy()
    
    p = r.copy()
    u = r.copy()
    q = np.zeros(N)

    res_history = [np.linalg.norm(r)]
    tol_abs = tol * res_history[0] if res_history[0] != 0 else tol 

    rho_old = np.dot(r, r0_star)

    for i in range(max_iter):
        if res_history[-1] < tol_abs:
            break

        if rho_old == 0:
            print("Breakdown in CGS: rho = 0")
            break

        p_aux = A @ p

        den_alpha = np.dot(p_aux, r0_star)
        if den_alpha == 0:
            print("Breakdown in CGS: den_alpha = 0")
            break

        alpha = rho_old / den_alpha

        q = u - alpha * p_aux
        u_plus_q = u + q

        x = x + alpha * u_plus_q
        r = r - alpha * A @ u_plus_q

        rho_new = np.dot(r, r0_star)
        beta = rho_new / rho_old

        u = r + beta * q 
        p = u + beta * (q + beta * p)

        rho_old = rho_new

        res_history.append(np.linalg.norm(r))

    return x, res_history