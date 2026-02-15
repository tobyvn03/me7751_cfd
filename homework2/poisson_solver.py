import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
File:   poisson_solver.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D Poisson equation using iterative methods (Jacobi, Gauss-Seidel, or SOR).
"""

def main():
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-n', type=int, default=10, help='Number of grid points')
    parser.add_argument('-m', '--method', type=str, default='jacobi', choices=['jacobi', 'gauss-seidel', 'sor'], help='Iterative method to use')
    parser.add_argument('-k', '--max-iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Relaxation factor for SOR method')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-6, help='Tolerance for convergence')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print intermediate results for the first 5 iterations')
    args = parser.parse_args()

    N = args.n
    method = str(args.method)
    k_max = args.max_iter
    alpha = args.alpha
    tolerance = args.tolerance
    verbose = args.verbose

    (x, y), h = create_grid(N)
    Q = 4*y**3 - 6*y**2 + 2 - 6*(1 - x**2)*(2*y - 1)
    T_initial = (2*y**3 - 3*y**2 + 1)
    T_analytical = (1 - x**2)*(2*y**3 - 3*y**2 + 1)

    print(f'Grid size: {N}x{N}, Method: {method}, Max iterations: {k_max}, Tolerance: {tolerance}, Alpha (for SOR): {alpha}')

    if method == 'jacobi':
        T, k = jacobi_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose)
    elif method == 'gauss-seidel':
        T, k = gauss_seidel_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose)
    elif method == 'sor':
        T, k = sor_method(N, x, y, h, Q, T_initial, k_max, alpha, tolerance, verbose)

    if verbose:
        print(f'Grid size: {N}x{N}, Method: {method}, Max iterations: {k_max}, Tolerance: {tolerance}, Alpha (for SOR): {alpha}')
    print(f'{method.capitalize()} method converged in {k+1} iterations.')

    plot_solution(x, y, T_initial, 'initial', output_file='initial_solution.png')
    plot_solution(x, y, T, method.capitalize(), output_file=f'{method}_solution.png')
    plot_solution(x, y, T_analytical, 'analytical', output_file='analytical_solution.png')
    plot_solution(x, y, Q, 'source term Q', output_file='source_term_Q.png')
    plot_solution(x, y, np.abs(T - T_analytical), 'absolute error', output_file='error.png')

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
        if verbose and k < 5:
            print(f'Iteration {k+1}:\n{T_new}')
        if np.linalg.norm(T_new - T) < tolerance:
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
        if verbose and k < 5:
            print(f'Iteration {k+1}:\n{T}')
        if np.linalg.norm(T - T_initial) < tolerance:
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
        if verbose and k < 5:
            print(f'Iteration {k+1}:\n{T}')
        if np.linalg.norm(T - T_initial) < tolerance:
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
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, T, levels=50, cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.title(f'Temperature Distribution: {method}')
    plt.xlabel('x')
    plt.ylabel('y')
    if output_file:
        plt.savefig(output_file)

if __name__ == '__main__':
    main()