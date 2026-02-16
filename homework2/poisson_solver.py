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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-n', nargs='+', type=int, default=10, help='Number of grid points')
    parser.add_argument('-m', '--method', type=str, default='jacobi', choices=['jacobi', 'gauss-seidel', 'sor'], help='Iterative method to use')
    parser.add_argument('-k', '--max-iter', nargs='+', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Relaxation factor for SOR method')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-6, help='Tolerance for convergence')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print intermediate results for the first 5 iterations')
    args = parser.parse_args()

    # Extract parameters from arguments
    N = args.n
    method = str(args.method)
    k_max = args.max_iter
    alpha = args.alpha
    tolerance = args.tolerance
    verbose = args.verbose

    # Plot vertical centerline results on Figure `f`
    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Centerline Temperature Distribution: {method.capitalize()} vs Analytical')
    ax.set_xlabel('y')
    ax.set_ylabel('Temperature')

    for Ni, k_max_i in zip(N, k_max):
        # Create grid and compute source term Q and T_initial
        (x, y), h = create_grid(Ni)
        Q = 4*y**3 - 6*y**2 + 2 - 6*(1 - x**2)*(2*y - 1)
        T_initial = (2*y**3 - 3*y**2 + 1)
        T_analytical = (1 - x**2)*(2*y**3 - 3*y**2 + 1)
        
        # Solve the Poisson equation using the specified method
        start_time = time.time()
        if method == 'jacobi':
            T, k = jacobi_method(Ni, x, y, h, Q, T_initial, k_max_i, tolerance, verbose)
        elif method == 'gauss-seidel':
            T, k = gauss_seidel_method(Ni, x, y, h, Q, T_initial, k_max_i, tolerance, verbose)
        elif method == 'sor':
            T, k = sor_method(Ni, x, y, h, Q, T_initial, k_max_i, alpha, tolerance, verbose)
        end_time = time.time()

        print(f'{Ni}, {method}, {k+1}, {end_time - start_time:.6f}, {np.mean(np.abs(T - T_analytical)):.6e}')

        # Plot contour results (each creates its own Figure `f` internally)
        plot_solution(x, y, T, method.capitalize(), output_file=f'results/{method}_solution_{Ni}.png')
        plot_solution(x, y, T_analytical, 'Analytical', output_file=f'results/analytical_solution_{Ni}.png')
        ax.plot(y[:, 0], T[:, Ni//2], label=f'N={Ni}, {method.capitalize()}')
        ax.plot(y[:, 0], T_analytical[:, Ni//2], label=f'N={Ni}, Analytical', linestyle='dashed')
    
    ax.legend()
    f.savefig(f'results/{method}_centerline.png')

def create_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    h = 1 / (N - 1)
    return np.meshgrid(x, y), h

def jacobi_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose):
    T = T_initial.copy()
    T_new = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    for k in range(k_max):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_new[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] + Q[i, j] * h**2)
        apply_boundary_conditions(x, y, T_new)
        diff = np.linalg.norm(T_new - T)
        if verbose:
            print(f'Iteration {k+1}: {diff:.6e}')
        if diff < tolerance:
            break
        T[:] = T_new[:]
    return T, k

def gauss_seidel_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose):
    T = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    for k in range(k_max):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] + Q[i, j] * h**2)
        apply_boundary_conditions(x, y, T)
        diff = np.linalg.norm(T - T_initial)
        if verbose:
            print(f'Iteration {k+1}: {diff:.6e}')
        if diff < tolerance:
            break
    return T, k

def sor_method(N, x, y, h, Q, T_initial, k_max, alpha, tolerance, verbose):
    T = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    for k in range(k_max):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T[i, j] = (1 - alpha) * T[i, j] + alpha * 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] + Q[i, j] * h**2)
        apply_boundary_conditions(x, y, T)
        diff = np.linalg.norm(T - T_initial)
        if verbose:
            print(f'Iteration {k+1}: {diff:.6e}')
        if diff < tolerance:
            break
    return T, k

def apply_boundary_conditions(x, y, T):
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]
    T[:, 0] = (1 - x[:, 0]**2)*(2*y[:, 0]**3 - 3*y[:, 0]**2 + 1)
    T[:, -1] = 0

    # T[0, :] = (1 - x[0, :]**2)*(2*y[0, :]**3 - 3*y[0, :]**2 + 1)
    # T[-1, :] = 0
    # T[:, 0] = T[:, 1]
    # T[:, -1] = T[:, -2]

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
    main()