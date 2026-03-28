import numpy as np
from lid_driven_flow import pressure_boundary_conditions as apply_boundary_conditions

def sor_method(N, h, Q, T_initial, k_max=10000, alpha=1.5, tolerance=1e-6, verbose=False):
    T = T_initial.copy()
    T_new = T_initial.copy()
    apply_boundary_conditions(T)
    apply_boundary_conditions(T_new)

    for k in range(k_max):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_new[i, j] = (1 - alpha)*T[i, j] + alpha*0.25 * (T[i+1, j] + T_new[i-1, j] + T[i, j+1] + T_new[i, j-1] - Q[i, j] * h**2)
        
        apply_boundary_conditions(T_new)
        res_grid = calculate_residual(N, h, Q, T_new)
        residual = np.linalg.norm(res_grid)

        if verbose:
            print(f'{k+1}, {residual:.6e}')
        if residual < tolerance:
            break
        
        T[:] = T_new[:]
    return T, k, res_grid

def calculate_residual(N, h, Q, T):
    res_grid = np.zeros((N, N))

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            res_grid[i, j] = Q[i, j] - (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*T[i, j]) / h**2

    return res_grid