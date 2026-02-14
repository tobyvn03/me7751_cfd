import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Poisson solver')
    parser.add_argument('-n', type=int, default=100, help='Number of grid points')
    parser.add_argument('-m', '--method', choices=['jacobi', 'gauss-seidel', 'sor'], default='jacobi', help='Iterative method to use')
    parser.add_argument('-k', '--max_iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Relaxation factor')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Extract parameters from arguments
    N = args.n
    method = str(args.method)
    k_max = args.max_iter
    alpha = args.alpha
    tolerance = args.tolerance
    verbose = args.verbose

    # Discretize the domain and set up the problem
    (x, y), h = create_grid(N)
    Q = 4*y**3 - 6*y**2 + 2 - 6*(1 - x**2)*(2*y - 1)
    T_initial = 2*y**3 - 3*y**2 + 1
    T_analytical = (1 - x**2)*(2*y**3 - 3*y**2 + 1)

    # Solve the Poisson equation using the specified method
    if method == 'jacobi':
        T, k = jacobi_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose)
    elif method == 'gauss-seidel':
        # T, k = gauss_seidel_method(N, x, y, h, Q, T_initial, k_max, verbose)
        pass
    elif method == 'sor':
        # T, k = sor_method(N, x, y, h, Q, T_initial, k_max, alpha, verbose)
        pass

    # Compute the maximum error compared to the analytical solution
    error = np.max(np.abs(T - T_analytical))
    if verbose:
        print(f'{method.capitalize()} method converged in {k} iterations with max error {error:.6e}')

    # Plot the results
    plot_solution(x, y, T, method, output_file=f'{method}_solution.png')
    plot_solution(x, y, T_analytical, 'analytical', output_file='analytical_solution.png')
    plot_solution(x, y, np.abs(T - T_analytical), 'error', output_file='error.png')
    plot_solution(x, y, T_initial, 'initial', output_file='initial_solution.png')
    plot_solution(x, y, Q, 'source', output_file='source_term_Q.png')

def create_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    h = 1 / (N - 1)
    return np.meshgrid(x, y), h

def jacobi_method(N, x, y, h, Q, T_initial, k_max, tolerance, verbose):
    T = T_initial.copy()
    apply_boundary_conditions(x, y, T)
    for k in range(k_max):
        # Update interior points using the Jacobi formula
        T_new = T.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                T_new[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - h**2 * Q[i, j])
        apply_boundary_conditions(x, y, T_new)

        # Compute the L-infinity norm of the change
        delta_T = np.max(np.abs(T_new - T))
        if verbose:
            print(f'Iteration {k+1}, L-infty norm of change: {delta_T:.6e}')
        if delta_T < tolerance:
            break # Convergence criterion
        T = T_new
    return T, k

def apply_boundary_conditions(x, y, T):
    T[0, :] = T[1, :] # Neumann BC at y=0
    T[-1, :] = T[-2, :] # Neumann BC at y=1
    T[:, 0] = 2*y[:, 0]**3 - 3*y[:, 0]**2 + 1 # Dirichlet BC at x=0
    T[:, -1] = 0 # Dirichlet BC at x=1
    return T

def plot_solution(x, y, T, method, output_file=None):
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, T, levels=50, cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.title(f'Temperature Distribution: {method.capitalize()}')
    plt.xlabel('x')
    plt.ylabel('y')
    if output_file:
        plt.savefig(output_file)

if __name__ == '__main__':
    main()