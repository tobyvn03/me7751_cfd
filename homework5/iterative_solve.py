import numpy as np

def sor_method(N, h, dt, Q, T_initial, k_max=100000, alpha=1.5, tolerance=1e-6, verbose=False):
    T = T_initial.copy()
    pressure_boundary_conditions(T)  # Ensure initial guess satisfies boundary conditions

    # Track previous pressure to stop when field stops changing
    T_prev = T.copy()

    for k in range(k_max):
        # Vectorized Red-Black SOR update (in-place on T)
        # Interior block (exclude ghost boundaries)
        T_interior = T[1:-1, 1:-1]
        Q_interior = Q[1:-1, 1:-1]

        # Precompute neighbor sum factor (uses current T values)
        # neigh = 0.25*(T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - Q*h^2)
        neigh = 0.25 * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - Q_interior * h**2)

        # Create red-black masks for the interior grid
        rows, cols = T_interior.shape
        idx_sum = (np.arange(rows)[:, None] + np.arange(cols)[None, :])
        red_mask = (idx_sum % 2 == 0)
        black_mask = ~red_mask

        # RED update (uses neighbors from current T)
        T_interior[red_mask] = (1 - alpha) * T_interior[red_mask] + alpha * neigh[red_mask]

        # Recompute neighbor sums because red points were updated
        neigh = 0.25 * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - Q_interior * h**2)

        # BLACK update (now uses updated red neighbors)
        T_interior[black_mask] = (1 - alpha) * T_interior[black_mask] + alpha * neigh[black_mask]

        # Apply boundary conditions and compute max change vs previous iteration
        pressure_boundary_conditions(T)
        dp_max = np.max(np.abs(T[1:-1, 1:-1] - T_prev[1:-1, 1:-1]))
        dp_dt_max = dp_max / dt

        if verbose:
            print(f'{k+1}, {dp_dt_max:.6e}')
        if dp_dt_max < tolerance:
            break

        # Prepare for next iteration
        T_prev[:] = T[:]

    return T, k, dp_dt_max

def pressure_boundary_conditions(p, couvette=False):
    # Apply dp/dn=0 (Neumann) boundary condition.
    # Left and right walls
    if not couvette:
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

    # Bottom and top walls
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

def calculate_residual(N, h, Q, T):
    res_grid = np.zeros((N, N))

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            res_grid[i, j] = Q[i, j] - (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*T[i, j]) / h**2

    return np.max(np.abs(res_grid))