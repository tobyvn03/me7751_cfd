import pandas as pd
import numpy as np
import argparse
import time
import os
from finite_differences import d_dx_central, d_dy_central
from plotting import *

"""
File:   solver.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D lid-driven cavity flow of a viscous,
        incompressible Newtonian fluid using Lattice Boltzmann Method
        on a D2Q9 lattice using the BGK collision operator.
"""

def main():
    # Initialize plot settings
    fig_u, axes_u = initialize_plot(xlabel='u', ylabel='y')
    fig_v, axes_v = initialize_plot(xlabel='x', ylabel='v')

    # Get lattice weights and velocity directions for D2Q9
    indexes = np.arange(9)
    Cx = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    Cy = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Write header to summary output
    with open(output_file, 'w') as file_out:
        file_out.write('trial,N,Re,U,t,tau,MAE,Time\n')
    
    for trial, N, Re, U in zip(trial_ids, N_values, Re_values, U_values):
        # Get coordinate grid at cell centers
        (x, y) = create_grid(N, N_ghost=1)

        # Get relaxation time from Reynolds number
        nu = U*N/Re
        tau = 3 * nu + 0.5
        if tau <= 0.5:
            print(f"Warning: tau={tau:.4f} is not greater than 0.5 for trial {trial}.")
            continue

        # Define ghost cell mask for applying boundary conditions
        ghost_mask = np.full((N+2, N+2), False)
        ghost_mask[0, :] = True
        ghost_mask[-1, :] = True
        ghost_mask[:, 0] = True
        ghost_mask[:, -1] = True

        # Initialize solution with ghost cells
        F = np.ones((N+2, N+2, 9)) + 0.01 * np.random.randn(N+2, N+2, 9)
        F[ghost_mask, :] = 0

        # Write header to verbose output
        if verbose:
            with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'w') as file_out:
                file_out.write('t      Ma         |du_dt|    |dv_dt|    |dp_dt|\n')

        # Solve the governing equations using the explicit scheme
        t = 0
        start_time = time.time()
        while True:
            # Free-streaming propagation step
            for i in indexes:
                F[:, :, i] = np.roll(F[:, :, i], Cx[i], axis=1)
                F[:, :, i] = np.roll(F[:, :, i], Cy[i], axis=0)

            # Compute macroscopic variables
            rho = np.sum(F, axis=2)
            u = np.sum(F * Cx, axis=2) / rho
            v = np.sum(F * Cy, axis=2) / rho       

            # Collision step with BGK operator
            F_eq = np.zeros(F.shape)
            for i in indexes:
                F_eq[:, :, i] = (rho * weights[i] * (1 + 3*(Cx[i]*u + Cy[i]*v) + 9/2 *(Cx[i]*u + Cy[i]*v)**2 - 3/2 * (u**2 + v**2)))
            F -= (F - F_eq) / tau

            # Apply boundary conditions to F at ghost cells
            F_ghost = F[ghost_mask, :]
            F_reflected = F_ghost[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
            F[ghost_mask, :] = F_reflected

            # Apply lid-driven boundary condition at top wall
            F[-1, :, 6] -= 6 * weights[6] * rho[-1, :] * Cx[6] * (2 * U - u[-2, :])
            F[-1, :, 8] -= 6 * weights[8] * rho[-1, :] * Cx[8] * (2 * U - u[-2, :])

            # Compute max time derivative for convergence check
            rho_new = np.sum(F, axis=2)
            u_new = np.sum(F * Cx, axis=2) / rho_new
            v_new = np.sum(F * Cy, axis=2) / rho_new    
            du_dt_max = np.max(np.abs(u_new[1:-1, 1:-1] - u[1:-1, 1:-1]))
            dv_dt_max = np.max(np.abs(v_new[1:-1, 1:-1] - v[1:-1, 1:-1]))
            dp_dt_max = np.max(np.abs(rho_new[1:-1, 1:-1] - rho[1:-1, 1:-1])) / 3

            # Write results to verbose output
            if verbose:
                speed = np.sqrt(u_new[1:-1, 1:-1]**2 + v_new[1:-1, 1:-1]**2)
                Ma = np.max(speed) * np.sqrt(3)
                with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'a') as file_out:
                    file_out.write(f'{t:<6} {Ma:<10.4e} {du_dt_max:<10.4e} {dv_dt_max:<10.4e} {dp_dt_max:<10.4e}\n')

            # Skip this trial if any of the time derivatives are NaN
            if np.isnan(du_dt_max) or np.isnan(dv_dt_max) or np.isnan(dp_dt_max):
                print(f"Warning: NaN detected in time derivatives at trial {trial}, time step {t}. Skipping this trial.")
                break

            # Graph intermediate heatmaps if requested
            if t % plot_every == 0 and heatmaps:
                divergence = d_dx_central(u_new, 1)[1:-1, 1:-1] + d_dy_central(v_new, 1)[1:-1, 1:-1]
                plot_heatmap(divergence, title=f'{plot_file}_divergence_trial{trial}')
                plot_heatmap(rho_new[1:-1, 1:-1] / 3, title=f'{plot_file}_pressure_trial{trial}')
                plot_heatmap(u_new[1:-1, 1:-1], title=f'{plot_file}_xvelocity_trial{trial}')
                plot_heatmap(v_new[1:-1, 1:-1], title=f'{plot_file}_yvelocity_trial{trial}')
            if t % plot_every == 0 and streamlines:
                plot_fluid_streamlines(u_new, v_new, x, y, title=f'{plot_file}_streamlines_trial{trial}')

            # Check for convergence and plot results if converged
            tolerance = 1e-12
            if du_dt_max < tolerance and dv_dt_max < tolerance and dp_dt_max < tolerance:
                divergence = d_dx_central(u_new, 1)[1:-1, 1:-1] + d_dy_central(v_new, 1)[1:-1, 1:-1]
                plot_heatmap(divergence, title=f'{plot_file}_divergence_trial{trial}')
                plot_heatmap(rho_new[1:-1, 1:-1] / 3, title=f'{plot_file}_pressure_trial{trial}')
                plot_heatmap(u_new[1:-1, 1:-1], title=f'{plot_file}_xvelocity_trial{trial}')
                plot_heatmap(v_new[1:-1, 1:-1], title=f'{plot_file}_yvelocity_trial{trial}')
                plot_fluid_streamlines(u_new, v_new, x, y, title=f'{plot_file}_streamlines_trial{trial}')
                break

            t += 1
        end_time = time.time()

        # Plot x-velocity profile
        y_interior = y[1:-1, 0]
        x_index = np.argmin(np.abs(x[0, :] - 1/2))
        u_interior = u_new[1:-1, x_index]
        axes_u.plot(u_interior, y_interior, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Plot y-velocity profile
        x_interior = x[0, 1:-1]
        y_index = np.argmin(np.abs(y[:,0] - 1/2))
        v_interior = v_new[y_index, 1:-1]
        axes_v.plot(x_interior, v_interior, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Plot analytical solution and compute MAE
        # (x_N129, y_N129), h = create_grid(129, N_ghost=1)
        # x_indices = np.array([1, 9, 10, 11, 13, 21, 30, 31, 65, 104, 111, 117, 122, 123, 124, 125, 129]) - 1
        # y_indices = np.array([1, 8, 9, 10, 14, 23, 37, 59, 65, 80, 95, 110, 123, 124, 125, 126, 129]) - 1
        # if Re == 100:
        #     v_analytical = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])
        #     u_analytical = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
        # elif Re == 400:
        #     v_analytical = np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000])
        #     u_analytical = np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000])
        # else:
        #     raise ValueError(f"Analytical solution not available for Re={Re}")
        # axes_v.plot(x_N129[0, x_indices], v_analytical, 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        # axes_u.plot(u_analytical, y_N129[y_indices, 0], 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        # MAE = compute_mae(u_new, v_new, x, y, u_analytical, v_analytical, y_N129[y_indices, 0], x_N129[0, x_indices])
        MAE = 0 # Temporary

        # Write results to summary file
        with open(output_file, 'a') as file_out:
            file_out.write(f"{trial},{N},{Re},{U},{t},{tau},{MAE},{end_time - start_time}\n")

    # Save output plots
    fig_u.savefig(plot_file + '_centerline_xvelocity.png')
    fig_v.savefig(plot_file + '_centerline_yvelocity.png')
    plt.close(fig_u)
    plt.close(fig_v)

def create_grid(N_interior, N_ghost=1):
    N = N_interior + 2 * N_ghost
    x = np.arange(N) + 0.5 - N_ghost
    y = np.arange(N) + 0.5 - N_ghost
    (X, Y) = np.meshgrid(x, y)
    return (X, Y)

# def propagate_periodic(F, Cx, Cy):
#     for i in range(len(Cx)):
#         F[:, :, i] = np.roll(F[:, :, i], Cx[i], axis=1)
#         F[:, :, i] = np.roll(F[:, :, i], Cy[i], axis=0)

# def propagate(F):
#     F_new = np.zeros_like(F)
#     F_new[:, :, 0] = F[:, :, 0] # Direction 0: center
#     F_new[:, :-1, 1] = F[:, 1:, 1] # Direction 1: right
#     F_new[:-1, :-1, 2] = F[1:, 1:, 2] # Direction 2: up-right
#     F_new[:-1, :, 3] = F[1:, :, 3] # Direction 3: up
#     F_new[:-1, 1:, 4] = F[1:, :-1, 4] # Direction 4: up-left
#     F_new[:, 1:, 5] = F[:, :-1, 5] # Direction 5: left
#     F_new[1:, 1:, 6] = F[:-1, :-1, 6] # Direction 6: down-left
#     F_new[1:, :, 7] = F[:-1, :, 7] # Direction 7: down
#     F_new[1:, :-1, 8] = F[:-1, 1:, 8] # Direction 8: down-right
#     return F_new

# def compute_macroscopic_variables(F, Cx, Cy):
#     # Compute macroscopic density and velocity from distribution functions
#     rho = np.sum(F, axis=2)
#     u = np.sum(F * Cx, axis=2) / rho
#     v = np.sum(F * Cy, axis=2) / rho
#     return rho, u, v

# def compute_equilibrium_distribution(F, rho, u, v, Cx, Cy, weights):
#     # Compute equilibrium distribution functions based on local density and velocity
#     F_eq = np.zeros_like(F)
#     idxs = np.arange(len(Cx))
#     for i, cx, cy, w in zip(idxs, Cx, Cy, weights):
#         F_eq[:, :, i] = (rho * w * (1 + 3*(cx*u + cy*v) + 9*(cx*u + cy*v)**2/2 - 3*(u**2 + v**2)/2))
#     # C_dot_u = Cx[None, None, :] * u[:, :, None] + Cy[None, None, :] * v[:, :, None]
#     # F_eq = weights[None, None, :] * rho[:, :, None] * (1 + C_dot_u / Cs_sq + 9/2 * C_dot_u**2 - 1/2 * (u[:, :, None]**2 + v[:, :, None]**2) / Cs_sq)
#     return F_eq

# def apply_boundary_conditions(F, U_lid, rho, weights):
#     # Bottom wall (y=-0.5): no-slip
#     F[:, 0, 3] = F[:, 0, 7]
#     F[:, 0, 2] = F[:, 0, 6]
#     F[:, 0, 4] = F[:, 0, 8]
#     # Top wall (y=N+0.5): moving lid with velocity U_lid
#     F[:, -1, 7] = F[:, -1, 3]
#     F[:, -1, 6] = F[:, -1, 2] + 6 * weights[2] * rho[:, -1] * U_lid
#     F[:, -1, 8] = F[:, -1, 4] - 6 * weights[2] * rho[:, -1] * U_lid
#     # Left wall (x=-0.5): no-slip
#     F[0, :, 1] = F[0, :, 5]
#     F[0, :, 2] = F[0, :, 6]
#     F[0, :, 8] = F[0, :, 4]
#     # Right wall (x=N+0.5): no-slip
#     F[-1, :, 5] = F[-1, :, 1]
#     F[-1, :, 6] = F[-1, :, 2]
#     F[-1, :, 4] = F[-1, :, 8]
#     return F

def compute_mae(u, v, x, y, analytical_u, analytical_v, analytical_y, analytical_x):
    h = x[0, 1] - x[0, 0]
    # Interpolate u at x=0.5, analytical_y
    u_x_grid = x[0, :-1] + h / 2
    u_y_grid = y[:, 0]
    u_interp = RegularGridInterpolator((u_y_grid, u_x_grid), u, method='linear')
    u_numerical = u_interp((analytical_y, 0.5 * np.ones_like(analytical_y)))
    # Interpolate v at analytical_x, y=0.5
    v_x_grid = x[0, :]
    v_y_grid = y[:-1, 0] + h / 2
    v_interp = RegularGridInterpolator((v_y_grid, v_x_grid), v, method='linear')
    v_numerical = v_interp((0.5 * np.ones_like(analytical_x), analytical_x))
    # Compute absolute errors
    u_errors = np.abs(u_numerical - analytical_u)
    v_errors = np.abs(v_numerical - analytical_v)
    # Mean absolute error across all points
    MAE = np.mean(np.concatenate([u_errors, v_errors]))
    return MAE

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input .csv file with parameters and trials')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output .csv file to save results')
    parser.add_argument('-p', '--plot_file', type=str, default=None, help='Output .png file to save results')
    parser.add_argument('-f', '--base_folder', type=str, default='./', help='Base folder for trials (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Write verbose output file')
    parser.add_argument('-m', '--maps', action='store_true', help='Graph intermediate heatmaps')
    parser.add_argument('-s', '--streamlines', action='store_true', help='Graph velocity streamlines')
    parser.add_argument('--plot_every', type=int, default=500, help='Frequency of plotting intermediate results (default: 500)')
    args = parser.parse_args()

    # Extract parameters from arguments
    base_folder = args.base_folder
    input_file = args.input_file
    input_file = os.path.join(base_folder, input_file)
    output_file = args.output_file
    output_file = os.path.join(base_folder, output_file)
    plot_file = args.plot_file
    plot_file = os.path.join(base_folder, plot_file)
    verbose = args.verbose
    heatmaps = args.maps
    streamlines = args.streamlines
    plot_every = args.plot_every

    # Extract input parameters from input_file
    df = pd.read_csv(input_file)
    trial_ids = df['trial'].tolist()
    N_values = df['N'].tolist()
    Re_values = df['Re'].tolist()
    U_values = df['U'].tolist()
    
    main()