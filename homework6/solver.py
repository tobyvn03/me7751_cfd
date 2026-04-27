import pandas as pd
import numpy as np
import argparse
import time
import os
from scipy.interpolate import RegularGridInterpolator
from finite_differences import d_dx_central, d_dy_central
from plotting import *

"""
File:   solver.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D lid-driven cavity flow of a viscous,
        incompressible Newtonian fluid using Lattice Boltzmann Method
        on a D2Q9 lattice using the BGK collision operator.
"""

# Get lattice weights and velocity directions for D2Q9
NUM_DIRECTIONS = 9
CS_SQUARED = 1/3
CX = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
CY = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
WEIGHTS = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

def main():
    # Initialize plot settings
    fig_u, axes_u = initialize_plot(xlabel='u', ylabel='y')
    fig_v, axes_v = initialize_plot(xlabel='x', ylabel='v')
    reference_profiles = None

    # Write header to summary output
    with open(output_file, 'w') as file_out:
        file_out.write('trial,N,Re,U,t,tau,MAE,Time\n')
    
    for trial, N, Re, U in zip(trial_ids, N_values, Re_values, U_values):
        # Get coordinate grid at cell centers
        (x, y) = create_grid(N)
        L = N-1

        # Get relaxation time from Reynolds number
        nu = U*L/Re
        tau = nu/CS_SQUARED + 0.5
        if tau <= 0.5:
            print(f"Warning: tau={tau:.4f} is not greater than 0.5 for trial {trial}. Skipping this trial.")
            continue
        if U >= np.sqrt(CS_SQUARED):
            print(f"Warning: U={U:.4f} is not less than the speed of sound {np.sqrt(CS_SQUARED):.4f} for trial {trial}. Skipping this trial.")
            continue

        # Initialize solution with equilibrium distribution for uniform density and zero velocity
        rho = np.ones((N, N))
        u = np.zeros((N, N))
        v = np.zeros((N, N))
        u[-1, :] = U
        F = compute_equilibrium_distribution(N, rho, u, v)

        # Write header to verbose output
        if verbose:
            with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'w') as file_out:
                file_out.write('t      Ma         |du_dt|    |dv_dt|\n')

        # Solve the governing equations using the explicit scheme
        t = 0
        start_time = time.time()
        while True:
            # Perform one time step of the Lattice Boltzmann Method
            rho, u, v = compute_macroscopic_variables(F, U) # Compute macroscopic variables from distribution functions
            F_eq = compute_equilibrium_distribution(N, rho, u, v)
            F -= (F - F_eq) / tau # Collision step with BGK operator
            F_new = propagate(F) # Free-streaming propagation step
            rho_new, _, _ = compute_macroscopic_variables(F_new, U)
            apply_boundary_conditions(F_new, U, rho_new)

            # Compute max time derivative for convergence check
            _, u_new, v_new = compute_macroscopic_variables(F_new, U)
            du_dt_max = np.max(np.abs(u_new - u))
            dv_dt_max = np.max(np.abs(v_new - v))

            # Write results to verbose output
            if verbose:
                speed = np.sqrt(u_new**2 + v_new**2)
                Ma = np.max(speed) / np.sqrt(CS_SQUARED)
                with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'a') as file_out:
                    file_out.write(f'{t:<6} {Ma:<10.4e} {du_dt_max:<10.4e} {dv_dt_max:<10.4e}\n')
                # print(f"min(rho) = {np.min(rho_new):.4e}")

            # Skip this trial if any of the time derivatives are NaN
            if np.isnan(du_dt_max) or np.isnan(dv_dt_max):
                print(f"Warning: NaN detected in time derivatives at trial {trial}, time step {t}. Skipping this trial.")
                break

            # Graph intermediate heatmaps if requested
            if t % plot_every == 0:
                divergence = d_dx_central(u_new, dx=1) + d_dy_central(v_new, dy=1)
                plot_heatmap(divergence, title=f'{plot_file}_divergence_trial{trial}')
                plot_heatmap(u_new, title=f'{plot_file}_xvelocity_trial{trial}')
                plot_heatmap(v_new, title=f'{plot_file}_yvelocity_trial{trial}')
                plot_fluid_streamlines(u_new/U, v_new/U, x/L, y/L, title=f'{plot_file}_streamlines_trial{trial}')

            # Check for convergence and plot results if converged
            tolerance = 1e-12
            if du_dt_max < tolerance and dv_dt_max < tolerance:
                divergence = d_dx_central(u_new, dx=1) + d_dy_central(v_new, dy=1)
                plot_heatmap(divergence, title=f'{plot_file}_divergence_trial{trial}')
                plot_heatmap(u_new, title=f'{plot_file}_xvelocity_trial{trial}')
                plot_heatmap(v_new, title=f'{plot_file}_yvelocity_trial{trial}')
                plot_fluid_streamlines(u_new/U, v_new/U, x/L, y/L, title=f'{plot_file}_streamlines_trial{trial}')
                break

            t += 1
            F[:] = F_new[:]
        end_time = time.time()

        # Plot x-velocity profile
        u_interior = u_new[:, N//2]
        axes_u.plot(u_interior/U, y[:, 0]/L, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Plot y-velocity profile
        v_interior = v_new[N//2, :]
        axes_v.plot(x[0, :]/L, v_interior/U, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Define and plot analytical solution for this trial
        if len(trial_ids) == 1:
            plot_analytical(Re, trial, axes_u, axes_v)

        # Use trial 1 as the reference solution for MAE on all trials.
        if trial == 1:
            reference_profiles = {
                'u': u_interior / U,
                'v': v_interior / U,
                'y': y[:, 0] / L,
                'x': x[0, :] / L,
            }
        elif reference_profiles is None:
            raise ValueError('Trial 1 must complete successfully before MAE can be computed for later trials.')

        MAE = compute_mae(
            u_new/U,
            v_new/U,
            x/L,
            y/L,
            reference_profiles['u'],
            reference_profiles['v'],
            reference_profiles['y'],
            reference_profiles['x'],
        )

        # Write results to summary file
        with open(output_file, 'a') as file_out:
            file_out.write(f"{trial},{N},{Re},{U},{t},{tau},{MAE},{end_time - start_time}\n")

    # Save output plots
    fig_u.savefig(plot_file + '_centerline_xvelocity.png')
    fig_v.savefig(plot_file + '_centerline_yvelocity.png')
    plt.close(fig_u)
    plt.close(fig_v)

def create_grid(N):
    x = np.arange(N)
    y = np.arange(N)
    (X, Y) = np.meshgrid(x, y)
    return (X, Y)

def propagate(F):
    F_new = np.zeros_like(F)
    F_new[:, :, 0] = F[:, :, 0] # Direction 0: center
    F_new[:, 1:, 1] = F[:, :-1, 1] # Direction 1: right
    F_new[1:, 1:, 2] = F[:-1, :-1, 2] # Direction 2: up-right
    F_new[1:, :, 3] = F[:-1, :, 3] # Direction 3: up
    F_new[1:, :-1, 4] = F[:-1, 1:, 4] # Direction 4: up-left
    F_new[:, :-1, 5] = F[:, 1:, 5] # Direction 5: left
    F_new[:-1, :-1, 6] = F[1:, 1:, 6] # Direction 6: down-left
    F_new[:-1, :, 7] = F[1:, :, 7] # Direction 7: down
    F_new[:-1, 1:, 8] = F[1:, :-1, 8] # Direction 8: down-right
    return F_new

def compute_macroscopic_variables(F, U_lid):
    # Compute macroscopic density and velocity from distribution functions
    rho = np.sum(F, axis=2)
    u = np.sum(F * CX, axis=2) / rho
    v = np.sum(F * CY, axis=2) / rho

    u[0, :] = 0
    u[-1, :] = U_lid
    u[:, 0] = 0
    u[:, -1] = 0

    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    return rho, u, v

def compute_equilibrium_distribution(N, rho, u, v):
    # Compute equilibrium distribution functions based on local density and velocity
    F_eq = np.zeros((N, N, NUM_DIRECTIONS))
    for i in range(NUM_DIRECTIONS):
        F_eq[:, :, i] = rho * WEIGHTS[i] * (1 + (CX[i]*u + CY[i]*v)/CS_SQUARED + (CX[i]*u + CY[i]*v)**2/2/CS_SQUARED**2 - (u**2 + v**2)/2/CS_SQUARED)
    return F_eq

def apply_boundary_conditions(F, U_lid, rho):
    # Bottom wall no-slip
    F[0, :, 3] = F[0, :, 7]
    F[0, :, 2] = F[0, :, 6]
    F[0, :, 4] = F[0, :, 8]
    # Top wall moving lid with velocity U_lid
    F[-1, :, 7] = F[-1, :, 3] - 2 * WEIGHTS[3] * rho[-1, :] * U_lid * CX[3] / CS_SQUARED
    F[-1, :, 6] = F[-1, :, 2] - 2 * WEIGHTS[2] * rho[-1, :] * U_lid * CX[2] / CS_SQUARED
    F[-1, :, 8] = F[-1, :, 4] - 2 * WEIGHTS[4] * rho[-1, :] * U_lid * CX[4] / CS_SQUARED
    # Left wall no-slip
    F[:, 0, 1] = F[:, 0, 5]
    F[:, 0, 2] = F[:, 0, 6]
    F[:, 0, 8] = F[:, 0, 4]
    # Right wall no-slip
    F[:, -1, 5] = F[:, -1, 1]
    F[:, -1, 6] = F[:, -1, 2]
    F[:, -1, 4] = F[:, -1, 8]

def compute_mae(u, v, x, y, analytical_u, analytical_v, analytical_y, analytical_x):
    # Numerical grid coordinates
    x_coords = x[0, :]
    y_coords = y[:, 0]

    # Determine centerline indices (assume square grid)
    N = u.shape[1]
    x_center = x_coords[N // 2]
    y_center = y_coords[N // 2]

    # Interpolators expect ordering (y, x)
    u_interp = RegularGridInterpolator((y_coords, x_coords), u, method='linear', bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((y_coords, x_coords), v, method='linear', bounds_error=False, fill_value=None)

    # Points where literature provides u: vertical centerline at x_center
    pts_u = np.column_stack((analytical_y, np.full_like(analytical_y, x_center)))
    # Points where literature provides v: horizontal centerline at y_center
    pts_v = np.column_stack((np.full_like(analytical_x, y_center), analytical_x))

    u_numerical = u_interp(pts_u)
    v_numerical = v_interp(pts_v)

    # Compute absolute errors only at literature points
    u_errors = np.abs(u_numerical - analytical_u)
    v_errors = np.abs(v_numerical - analytical_v)

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
    plot_every = args.plot_every

    # Extract input parameters from input_file
    df = pd.read_csv(input_file)
    trial_ids = df['trial'].tolist()
    N_values = df['N'].tolist()
    Re_values = df['Re'].tolist()
    U_values = df['U'].tolist()
    
    main()