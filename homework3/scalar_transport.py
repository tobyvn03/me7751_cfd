import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

"""
File:   scalar_transport.py
Author: Toby Viet Nguyen
Desc:   This script solves the 1-D unsteady transport equation ∂φ/∂t + U ∂φ/∂x = ∂/∂x (Γ ∂φ/∂x)
        given constant U, Γ using an explicit, implicit, and Crank-Nicolson scheme.
"""

def main():
    # Plot vertical centerline results
    if plots_folder:
        f, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Centerline Temperature Distribution: {method.capitalize()} vs. Analytical', fontsize=16)
        ax.set_xlabel('y', fontsize=14)
        ax.set_ylabel('Temperature', fontsize=14)
        ax.tick_params(labelsize=12)

    for N_i, U_i, G_i, t_final_i, t_start_i, dt_i in zip(N, U, G, t_final, t_start, dt):
        # Create grid and compute source term Q and T_initial
        x, h = create_grid(N_i)
        phi_initial = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-10)**2)

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
            T, k, _ = multi_grid_method(Ni, x, y, h, Q, T_initial, int(1e5), 5e-2, verbose=verbose)
        end_time = time.time()

        # Compute analytical solution and MAE
        if U_i == 1 and G_i == 0.01:
            phi_analytical = (4*np.pi*G_i*t_final_i)**(-0.5) * np.exp(-(x-U_i*t_final_i)**2/(4*G_i*t_final_i))
            mae = np.mean(np.abs(phi_analytical - phi_initial)) # TODO replace phi_initial with numerical solution at t_final_i
        if U_i == 1 and G_i == 0:
            phi_analytical = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-U_i*t_final_i)**2)

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

def create_grid(N):
    h = 1 / (N - 1)
    x = np.linspace(0, 1, N)
    return x, h



if __name__ == "__main__":
    # Parse command-line arguments for simulation
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-n', '--number_grid_points', nargs='+', type=int, help='Number of grid points')
    parser.add_argument('-u', '--advective_velocity', nargs='+', type=float, default=1.0, help='Advection velocity U')
    parser.add_argument('-g', '--diffusivity', nargs='+', type=float, default=0.01, help='Diffusivity Γ')
    parser.add_argument('-T', '--final_time', nargs='+', type=float, default=40.0, help='Final time for unsteady simulation')
    parser.add_argument('-t', '--start_time', nargs='+', type=float, default=10.0, help='Start time for unsteady simulation')
    parser.add_argument('-d', '--time_step', nargs='+', type=float, default=0.01, help='Time step for unsteady simulation')
    parser.add_argument('-s', '--scheme', nargs='+', type=str, default='explicit', choices=['explicit', 'implicit', 'crank-nicolson'], help='Explicit, implicit, or Crank-Nicolson scheme')

    # Parse command-line arguments for plotting
    parser.add_argument('--plot_x', nargs='+', type=float, help='X values for plotting')
    parser.add_argument('--plot_t', nargs='+', type=float, help='T values for plotting')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output folder for the plots (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output for debugging')
    args = parser.parse_args()

    # Extract parameters from arguments
    N = args.number_grid_points
    U = args.advective_velocity
    G = args.diffusivity
    t_final = args.final_time
    t_start = args.start_time
    dt = args.time_step
    scheme = str(args.scheme)
    
    plot_x = args.plot_x
    plot_t = args.plot_t
    plots_folder = args.output
    verbose = args.verbose

    main()