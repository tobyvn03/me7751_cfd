import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

"""
File:   poisson_solver.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D Poisson equation using iterative methods (Jacobi, Gauss-Seidel, or SOR).
"""

def main():
    # Plot vertical centerline results
    if plots_folder:
        f, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Centerline Temperature Distribution: {method.capitalize()} vs. Analytical', fontsize=16)
        ax.set_xlabel('y', fontsize=14)
        ax.set_ylabel('Temperature', fontsize=14)
        ax.tick_params(labelsize=12)

    for Ni, alpha_i in zip(N, alpha):
        # Create grid and compute source term Q and T_initial
        (x, y), h = create_grid(Ni, ghost_cells)
        Q = -(4*y**3 - 6*y**2 + 2 - 6*(1 - x**2)*(2*y - 1))
        T_initial = (2*y**3 - 3*y**2 + 1)
        T_analytical = (1 - x**2)*(2*y**3 - 3*y**2 + 1)

        # Solve the Poisson equation using the specified method
        start_time = time.time()
        if method == 'jacobi':
            T, k, _ = jacobi_method(Ni, x, y, h, Q, T_initial, int(1e5), 1e-2, verbose=verbose)
        elif method == 'gauss-seidel':
            if ghost_cells:
                raise NotImplementedError("Gauss-Seidel method with ghost cells is not implemented.")
            T, k, _ = gauss_seidel_method(Ni, x, y, h, Q, T_initial, int(1e5), 1e-2, verbose=verbose)
        elif method == 'sor':
            if ghost_cells:
                raise NotImplementedError("SOR method with ghost cells is not implemented.")
            T, k, _ = sor_method(Ni, x, y, h, Q, T_initial, int(1e5), alpha_i, 1e-2, verbose=verbose)
        elif method == 'multi-grid':
            if ghost_cells:
                raise NotImplementedError("Multi-grid method with ghost cells is not implemented.")
            T, k, _ = multi_grid_method(Ni, x, y, h, Q, T_initial, int(10), 1e-2, verbose=verbose)
        end_time = time.time()

        # Print residuals if verbose enabled, otherwise print results
        if not verbose:
            print(f'{Ni}, {method}, {k+1}, {end_time - start_time:.6f}, {np.mean(np.abs(T - T_analytical)):.6e}, {alpha_i}, {h}')

        # Plot contour results
        if plots_folder:
            plot_solution(x, y, T, method.capitalize(), output_file=f'{plots_folder}/{method}_{Ni}.png')
            plot_solution(x, y, T_analytical, 'Analytical', output_file=f'{plots_folder}/analytical_{Ni}.png')
            # Plot centerline for the first 3 grid sizes
            if Ni in N[:3]:
                ax.plot(y[:, 0], T[:, Ni//2], label=f'N={Ni}, {method.capitalize()}')
                ax.plot(y[:, 0], T_analytical[:, Ni//2], label=f'N={Ni}, Analytical', linestyle='dashed')
    
    # Save vertical centerline plot
    if plots_folder:
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        f.savefig(f'{plots_folder}/{method}_centerline.png')

def create_grid(N, ghost_cells=False):
    h = 1 / (N - 1)
    x = np.linspace(0, 1, N)
    if ghost_cells:
        y = np.linspace(-h, 1 + h, N+2)
    else:
        y = np.linspace(0, 1, N)
    return np.meshgrid(x, y), h

def calculate_residual(N, h, Q, T):
    res_grid = np.zeros((N, N))

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            res_grid[i, j] = Q[i, j] - (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*T[i, j]) / h**2
    residual = np.linalg.norm(res_grid)

    return res_grid

def jacobi_method(N, x, y, h, Q, T_initial, k_max, tolerance, ghost_cells=False, verbose=False):
    T = T_initial.copy()
    T_new = T_initial.copy()
    apply_boundary_conditions(x, y, T, ghost_cells=ghost_cells)

    if ghost_cells:
        range_i = range(1, N + 1)
    else:
        range_i = range(1, N - 1)

    for k in range(k_max):
        for i in range_i:
            for j in range(1, N - 1):
                T_new[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - Q[i, j] * h**2)
        
        apply_boundary_conditions(x, y, T_new, ghost_cells=ghost_cells)
        res_grid = calculate_residual(N, h, Q, T_new)
        residual = np.linalg.norm(res_grid)

        if verbose:
            print(f'{k+1}, {residual:.6e}')
        if residual < tolerance:
            break
        
        T[:] = T_new[:]
    return T, k, res_grid

def gauss_seidel_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose=False):
    T = T_initial.copy()
    T_new = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    apply_boundary_conditions(x, y, T_new)

    for k in range(k_max):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_new[i, j] = 0.25 * (T[i+1, j] + T_new[i-1, j] + T[i, j+1] + T_new[i, j-1] - Q[i, j] * h**2)
        
        apply_boundary_conditions(x, y, T_new)
        res_grid = calculate_residual(N, h, Q, T_new)
        residual = np.linalg.norm(res_grid)

        if verbose:
            print(f'{k+1}, {residual:.6e}')
        if residual < tolerance:
            break
        
        T[:] = T_new[:]
    return T, k, res_grid

def sor_method(N, x, y, h, Q, T_initial, k_max, alpha, tolerance, verbose=False):
    T = T_initial.copy()
    T_new = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    apply_boundary_conditions(x, y, T_new)

    for k in range(k_max):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_new[i, j] = (1 - alpha)*T[i, j] + alpha*0.25 * (T[i+1, j] + T_new[i-1, j] + T[i, j+1] + T_new[i, j-1] - Q[i, j] * h**2)
        
        apply_boundary_conditions(x, y, T_new)
        res_grid = calculate_residual(N, h, Q, T_new)
        residual = np.linalg.norm(res_grid)

        if verbose:
            print(f'{k+1}, {residual:.6e}')
        if residual < tolerance:
            break
        
        T[:] = T_new[:]
    return T, k, res_grid

def multi_grid_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose=False):
    T = T_initial.copy()
    T_new = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    apply_boundary_conditions(x, y, T_new)
    
    # Create coarse grid
    N_coarse = N//2 + 1
    (x_coarse, y_coarse), h_coarse = create_grid(N_coarse)

    for k in range(k_max):
        # Smoothing using Gauss-Seidel method with 1 iteration
        T_fine, _, res_fine = gauss_seidel_method(N, x, y, h, Q, T, int(1), 1e-2, verbose=False)

        # Restriction
        res_coarse = restriction_operator(N, N_coarse, res_fine)
        print(f'res_coarse({res_coarse.shape}) =\n{res_coarse}')
        
        # Adapting SOR to solve for A@error = res instead of A@T = Q
        error_coarse, iters, _ = gauss_seidel_method(N_coarse, x_coarse, y_coarse, h_coarse, res_coarse, np.zeros((N_coarse, N_coarse)), int(1e3), 1e-2, verbose=False)
        print(f'error_coarse({error_coarse.shape}) =\n{error_coarse}')
        
        # Prolongation
        error_fine = prolongation_operator(N, N_coarse, error_coarse)
        print(f'error_fine({error_fine.shape}) =\n{error_fine}')
        T_new = T_fine + error_fine

        apply_boundary_conditions(x, y, T_new)
        res_grid = calculate_residual(N, h, Q, T_new)
        residual = np.linalg.norm(res_grid)

        if verbose:
            print(f'{k+1}, {iters}, {residual:.6e}')
        if residual < tolerance:
            break

        T[:] = T_new[:]
    return T, k, res_grid

def restriction_operator(N_fine, N_coarse, res_fine):
    res_coarse = np.zeros((N_coarse, N_coarse))

    # Center weighted by 4/16, each edge weighted by 2/16, each corner weighted by 1/16
    res_coarse[1:-1, 1:-1] = (4 * res_fine[2:-2:2, 2:-2:2] +
        2 * (res_fine[1:-3:2, 2:-2:2] + res_fine[3:-1:2, 2:-2:2] + 
            res_fine[2:-2:2, 1:-3:2] + res_fine[2:-2:2, 3:-1:2]) +
        1 * (res_fine[1:-3:2, 1:-3:2] + res_fine[1:-3:2, 3:-1:2] + 
            res_fine[3:-1:2, 1:-3:2] + res_fine[3:-1:2, 3:-1:2])) / 16

    return res_coarse

def prolongation_operator(N_fine, N_coarse, error_coarse):
    error_fine = np.zeros((N_fine, N_fine))

    # Fine grid directly inherits from coarse grid, then interpolate points in between
    error_fine[::2, ::2] = error_coarse
    error_fine[::2, 1::2] = 0.5 * (error_fine[::2, :-1:2] + error_fine[::2, 2::2])
    error_fine[1::2, ::2] = 0.5 * (error_fine[:-1:2, ::2] + error_fine[2::2, ::2])
    error_fine[1::2, 1::2] = 0.25 * (error_fine[:-1:2, :-1:2] + error_fine[:-1:2, 2::2] + 
                                error_fine[2::2, :-1:2] + error_fine[2::2, 2::2])

    return error_fine

def apply_boundary_conditions(x, y, T, ghost_cells=False):
    if ghost_cells:
        T[0, :] = T[2, :]
        T[-1, :] = T[-3, :]
        T[:, 0] = (1 - x[:, 0]**2)*(2*y[:, 0]**3 - 3*y[:, 0]**2 + 1)
        T[:, -1] = 0
    else:
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]
        T[:, 0] = (1 - x[:, 0]**2)*(2*y[:, 0]**3 - 3*y[:, 0]**2 + 1)
        T[:, -1] = 0

def plot_solution(x, y, T, method, output_file=None):
    f = plt.figure(figsize=(8, 6))
    plt.contourf(x, y, T, levels=50, cmap='viridis')
    cb = plt.colorbar()
    cb.set_label('Temperature', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.title(f'Temperature Distribution: {method}', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-n', nargs='+', type=int, default=10, help='Number of grid points')
    parser.add_argument('-m', '--method', type=str, default='jacobi', choices=['jacobi', 'gauss-seidel', 'sor', 'multi-grid'], help='Iterative method to use')
    parser.add_argument('-a', '--alpha', nargs='+', type=float, default=1.0, help='Relaxation factor for SOR method')
    parser.add_argument('-g', '--ghost_cells', action='store_true', help='Use 1 ghost row on top and bottom boundaries')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output folder for the plots (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print residuals at each iteration')
    args = parser.parse_args()

    # Extract parameters from arguments
    N = args.n
    method = str(args.method)
    alpha = args.alpha
    ghost_cells = args.ghost_cells
    plots_folder = args.output
    verbose = args.verbose

    main()