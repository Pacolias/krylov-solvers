import numpy as np 

def gmres(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Standard (not preconditioned) GMRES Method
    """

    N = len(b)

    if x0 is None:
        x = np.zeros(N)
    
    x = x0.copy()

    # 1. Initialization
    r0 = b - A @ x 
    beta = np.linalg.norm(r0)

    res_history = [beta]
    tol_abs = tol * beta if beta != 0 else tol 

    if beta < tol_abs:
        return x, res_history
    
    # We reserve memory for the V (Krylov basis) and H (Hessenberg) matrices
    V = np.zeros((N, max_iter + 1))
    H = np.zeros((max_iter + 1, max_iter))

    # Vectors to store the sines and cosines of Givens rotations
    cs = np.zeros(max_iter)
    sn = np.zeros(max_iter)

    # Vector xi containing the progressive update of the residue
    xi = np.zeros(max_iter + 1)

    V[:, 0] = r0 / beta 
    xi[0] =  beta

    # 2. Main loop
    for n in range(max_iter):
        # w_{n+1} = A * v_n
        w = A @ V[:, n]

        # Modified Gram-Schmidt
        for i in range(n + 1):
            H[i, n] = np.dot(w, V[:, i])
            w = w - H[i, n] * V[:, i]

        H[n+1, n] = np.linalg.norm(w)

        if H[n+1,n] != 0:
            V[:, n+1] = w / H[n+1, n]

        # Apply the Givens rotations calculated in previous iterations to the new column
        for i in range(n):
            temp_Hin = cs[i] * H[i, n] + sn[i] * H[i+1, n]
            H[i+1, n] = -sn[i] * H[i, n] + cs[i] * H[i+1, n]
            H[i, n] = temp_Hin

        # Calculate the new Givens rotation to cancel the subdiagonal element
        rad = np.sqrt(H[n, n]**2 + H[n+1, n]**2)
        if rad != 0:
            cs[n] = H[n, n] / rad 
            sn[n] = H[n+1, n] / rad 
        else:
            cs[n], sn[n] = 1.0, 1.0 
        
        # Update: Apply the new rotation to the diagonal element and cancel the subdiagonal.
        H[n, n] = cs[n] * H[n, n] + sn[n] * H[n+1, n]
        H[n+1, n] = 0.0

        # Update vecto xi
        # We must save xi[n+1] before overwriting xi[n]
        xi[n+1] = -sn[n] * xi[n]
        xi[n]   = cs[n] * xi[n]

        # Check convergence with the residue norm |xi_{n+1}|
        current_res = abs(xi[n+1])
        res_history.append(current_res)

        if current_res < tol_abs:
            # Solve the upper triangular system \tilde{H} y = xi
            y = np.linalg.solve(H[:n+1, :n+1], xi[:n+1])
            # x_n = x_0 + V * y
            x = x0 + V[:, :n+1] @ y 
            break 
    
    # If max_iter is reached without converging, we calculate the solution anyway
    if res_history[-1] >= tol_abs:
        y = np.linalg.solve(H[:max_iter, :max_iter], xi[:max_iter])
        x = x0 + V[:, :max_iter] @ y

    return x, res_history