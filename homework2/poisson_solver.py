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
    parser.add_argument('-v', '--verbose', action='store_true', help='Print intermediate results for the first 5 iterations')
    args = parser.parse_args()

    N = args.n
    method = str(args.method)
    k_max = args.max_iter
    alpha = args.alpha
    verbose = args.verbose

    (x, y), h = create_grid(N)
    Q = 4*y**3 - 6*y**2 + 2 - 6*(1 - x**2)*(2*y - 1)
    T_initial = (1 - x**2)*(2*y**3 - 3*y**2 + 1)
    T_analytical = (1 - x**2)*(2*y**3 - 3*y**2 + 1)

    plot_solution(x, y, T_initial, 'initial', output_file='initial_solution.png')

    if method == 'jacobi':
        T, k = jacobi_method(x, y, h, Q, T_initial, k_max, verbose)
    elif method == 'gauss-seidel':
        # T = gauss_seidel_method(x, y, h, Q, T_initial, k_max)
        pass
    elif method == 'sor':
        # T = sor_method(x, y, h, Q, T_initial, k_max, alpha)
        pass

    if verbose:
        print(f'{method.capitalize()} method converged in {k} iterations.')

    plot_solution(x, y, T, method.capitalize(), output_file=f'{method}_solution.png')
    plot_solution(x, y, T_analytical, 'analytical', output_file='analytical_solution.png')
    plot_solution(x, y, Q, 'source term Q', output_file='source_term_Q.png')
    plot_solution(x, y, T - T_analytical, 'error', output_file='error.png')

def create_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    h = 1 / (N - 1)
    return np.meshgrid(x, y), h

def jacobi_method(x, y, h, Q, T_initial, k_max, verbose):
    T_old = T_initial.copy()
    apply_boundary_conditions(x, y, T_old)
    for k in range(k_max):
        neighbors = (np.pad(T_old, ((0, 0), (1, 0)), mode='constant')[:, :-1]
                 + np.pad(T_old, ((0, 0), (0, 1)), mode='constant')[:, 1:]
                 + np.pad(T_old, ((1, 0), (0, 0)), mode='constant')[:-1, :]
                 + np.pad(T_old, ((0, 1), (0, 0)), mode='constant')[1:, :])
        T_new = 0.25 * (neighbors - Q * h**2)
        apply_boundary_conditions(x, y, T_new)

        change = np.linalg.norm(T_new - T_old, ord=np.inf)
        if verbose:
            print(f'Iteration {k+1}, max change: {change:.6e}')
        if change < 1e-6:
            break
        T_old = T_new
    return T_old, k

def apply_boundary_conditions(x, y, T):
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]
    T[:, 0] = (1 - x[:, 0]**2)*(2*y[:, 0]**3 - 3*y[:, 0]**2 + 1)
    T[:, -1] = 0

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