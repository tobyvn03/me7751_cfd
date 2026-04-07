import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os
from scipy.interpolate import RegularGridInterpolator
from finite_differences import *
from plotting import *
from iterative_solve import sor_method, pressure_boundary_conditions

"""
File:   lid_driven_flow.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D lid-driven cavity flow of a viscous,
        incompressible Newtonian fluid using an explicit projection method.
"""

def main():
    # Initialize plot settings
    fig_u, axes_u = initialize_plot(xlabel='u', ylabel='y')
    fig_v, axes_v = initialize_plot(xlabel='x', ylabel='v')
        
    # Write header to summary output
    with open(output_file, 'w') as file_out:
        file_out.write('trial,N,dt,Re,alpha,t,h,MAE,Time\n')
    
    for trial, N, dt, Re, alpha in zip(trial_ids, N_values, dt_values, Re_values, alpha_values):
        # Get coordinate grid at cell centers
        (x, y), h = create_grid(N, H=1, ghost_cells=True)
        nu = 1/Re

        # Initialize solution with ghost cells
        p = np.zeros((N + 2, N + 2))
        u = np.zeros((N + 2, N + 1))
        v = np.zeros((N + 1, N + 2))
        u_star = u.copy()
        v_star = v.copy()
        u_new = u.copy()
        v_new = v.copy()
        pressure_boundary_conditions(p)
        velocity_boundary_conditions(u, v, u_lid=1.0)

        # Write header to verbose output
        if verbose:
            with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'w') as file_out:
                file_out.write('t        CFL_adv    |du/dt|    |dv/dt|    |dp/dt|    k\n')

        # Solve the governing equations using the explicit scheme
        t = 0.0
        step = 0
        start_time = time.time()
        while True:
            # Compute intermediate change in velocity fields G and H
            G = calculate_G(u, v, nu, h, h)
            H = calculate_H(u, v, nu, h, h)

            # Compute intermediate velocity fields u_star and v_star
            u_star[1:-1, 1:] = u[1:-1, 1:] + G[1:-1, 1:] * dt
            v_star[1:, 1:-1] = v[1:, 1:-1] + H[1:, 1:-1] * dt
            velocity_boundary_conditions(u_star, v_star, u_lid=1.0)

            # Compute divergence of intermediate velocity field
            u_padded = np.hstack([u_star, np.zeros((u_star.shape[0], 1))]) # Pad last column of u_star with zeros to match p-shape (N+2, N+2)
            v_padded = np.vstack([v_star, np.zeros((1, v_star.shape[1]))]) # Pad last row of v_star with zeros to match p-shape (N+2, N+2)
            du_star_dx = d_dx_backward(u_padded, h)
            dv_star_dy = d_dy_backward(v_padded, h)
            div_star = du_star_dx + dv_star_dy

            # Solve Poisson equation for pressure correction using SOR method
            # For the projection step we need ∇²p = (1/dt)∇·u*.
            p, k, dp_dt_max = sor_method(N + 2, h, dt, div_star / dt, p, alpha=alpha)
            p -= np.mean(p) # Remove Neumann nullspace offset by zero-centering pressure

            # Update velocity fields using pressure correction
            u_new[1:-1, 1:] = u_star[1:-1, 1:] - d_dx_forward(p, h)[1:-1, 1:-1] * dt
            v_new[1:, 1:-1] = v_star[1:, 1:-1] - d_dy_forward(p, h)[1:-1, 1:-1] * dt
            velocity_boundary_conditions(u_new, v_new, u_lid=1.0)

            # Compute max time derivative for convergence check
            du_dt_max = np.max(np.abs(u_new[1:-1, 1:] - u[1:-1, 1:]) / dt)
            dv_dt_max = np.max(np.abs(v_new[1:, 1:-1] - v[1:, 1:-1]) / dt)

            # Write results to verbose output
            if verbose:
                speed = np.sqrt(u_new[1:-1, 1:]**2 + v_new[1:, 1:-1]**2)
                speed_max = np.max(speed)
                CFL = speed_max * dt / h
                with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'a') as file_out:
                    file_out.write(f'{t:<8.4f} {CFL:<10.4e} {du_dt_max:<10.4e} {dv_dt_max:<10.4e} {dp_dt_max:<10.4e} {k:<6}\n')

            # Skip plotting if any of the time derivatives are NaN
            if np.isnan(du_dt_max) or np.isnan(dv_dt_max):
                continue

            # Graph intermediate heatmaps if requested
            if step % 500 == 0 and heatmaps:
                divergence = d_dx_forward(u, h)[1:-1, 1:] + d_dy_forward(v, h)[1:, 1:-1]
                plot_heatmap(divergence[1:,1:], title=f'{plot_file}_divergence_trial{trial}')
                plot_heatmap(p[1:-1, 1:-1], title=f'{plot_file}_pressure_trial{trial}')
                plot_heatmap(u[1:-1, 1:], title=f'{plot_file}_xvelocity_trial{trial}')
                plot_heatmap(v[1:, 1:-1], title=f'{plot_file}_yvelocity_trial{trial}')
            if step % 500 == 0 and streamlines:
                plot_fluid_streamlines(u, v, x, y, title=f'{plot_file}_streamlines_trial{trial}')

            # Check for convergence and plot results if converged
            tolerance = 1e-6
            if du_dt_max < tolerance and dv_dt_max < tolerance:
                divergence = d_dx_forward(u, h)[1:-1, 1:] + d_dy_forward(v, h)[1:, 1:-1]
                plot_heatmap(divergence[1:,1:], title=f'{plot_file}_divergence_trial{trial}')
                plot_heatmap(p[1:-1, 1:-1], title=f'{plot_file}_pressure_trial{trial}')
                plot_heatmap(u[1:-1, 1:], title=f'{plot_file}_xvelocity_trial{trial}')
                plot_heatmap(v[1:, 1:-1], title=f'{plot_file}_yvelocity_trial{trial}')
                plot_fluid_streamlines(u, v, x, y, title=f'{plot_file}_streamlines_trial{trial}')
                break

            t += dt
            step += 1
            u[:] = u_new[:]
            v[:] = v_new[:]
        end_time = time.time()

        # Plot y-velocity profile
        x_interior = x[0, 1:-1]
        y_index = np.argmin(np.abs(y[:,0] - 1/2))
        v_interior = v[y_index, 1:-1]
        axes_v.plot(x_interior, v_interior, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Plot x-velocity profile
        y_interior = y[1:-1, 0]
        x_index = np.argmin(np.abs(x[0, :] - 1/2))
        u_interior = u[1:-1, x_index]
        axes_u.plot(u_interior, y_interior, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Plot analytical solution and compute MAE
        (x_N129, y_N129), h = create_grid(129, H=1, ghost_cells=True)
        x_indices = np.array([1, 9, 10, 11, 13, 21, 30, 31, 65, 104, 111, 117, 122, 123, 124, 125, 129]) - 1
        y_indices = np.array([1, 8, 9, 10, 14, 23, 37, 59, 65, 80, 95, 110, 123, 124, 125, 126, 129]) - 1
        if Re == 100:
            v_analytical = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])
            u_analytical = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
        elif Re == 400:
            v_analytical = np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000])
            u_analytical = np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000])
        else:
            raise ValueError(f"Analytical solution not available for Re={Re}")
        axes_v.plot(x_N129[0, x_indices], v_analytical, 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        axes_u.plot(u_analytical, y_N129[y_indices, 0], 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        MAE = compute_mae(u_new, v_new, x, y, u_analytical, v_analytical, y_N129[y_indices, 0], x_N129[0, x_indices])

        # Write results to summary file
        with open(output_file, 'a') as file_out:
            file_out.write(f"{trial},{N},{dt},{Re},{alpha},{t},{h},{MAE},{end_time - start_time}\n")

    # Save output plots
    fig_u.savefig(plot_file + '_centerline_xvelocity.png')
    fig_v.savefig(plot_file + '_centerline_yvelocity.png')

def create_grid(N, H, ghost_cells=False):
    h = H / N
    if ghost_cells:
        x = np.linspace(-h/2, H + h/2, N + 2)
        y = np.linspace(-h/2, H + h/2, N + 2)
    else:
        x = np.linspace(h/2, H - h/2, N)
        y = np.linspace(h/2, H - h/2, N)
    return np.meshgrid(x, y), h

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

def calculate_G(u, v, nu, dx, dy):
    dTxx_dx = 2 * nu * d2_dx2(u, dx)
    
    dv_dx = d_dx_forward(v, dx)[:, :-1] # Chop off last column of dv_dx
    dv_dx_n = np.vstack([dv_dx, np.zeros((1, dv_dx.shape[1]))]) # Pad last row with zeros to match u-shape (N+2, N+1)
    dv_dx_s = np.vstack([np.zeros((1, dv_dx.shape[1])), dv_dx]) # Pad first row with zeros

    Txy_n = nu * (d_dy_forward(u, dy) + dv_dx_n) 
    Txy_s = nu * (d_dy_backward(u, dy) + dv_dx_s)
    dTxy_dy = (Txy_n - Txy_s) / dy

    u_e = avg_x_forward(u)
    u_w = avg_x_backward(u)
    duu_dx = (u_e**2 - u_w**2) / dx

    u_n = avg_y_forward(u)
    u_s = avg_y_backward(u)

    avg_v_x = avg_x_forward(v)[:, :-1] # Chop off last column of avg_v_x
    v_n = np.vstack([avg_v_x, np.zeros((1, avg_v_x.shape[1]))]) # Pad last row with zeros to match u-shape (N+2, N+1)
    v_s = np.vstack([np.zeros((1, avg_v_x.shape[1])), avg_v_x]) # Pad first row with zeros

    duv_dy = (u_n * v_n - u_s * v_s) / dy

    G = -duu_dx - duv_dy + dTxx_dx + dTxy_dy

    return G

def calculate_H(u, v, nu, dx, dy):
    dTyy_dy = 2 * nu * d2_dy2(v, dy)

    du_dy = d_dy_forward(u, dy)[:-1, :] # Chop off last row of du_dy
    du_dy_e = np.hstack([du_dy, np.zeros((du_dy.shape[0], 1))]) # Pad last column with zeros to match v-shape (N+1, N+2)
    du_dy_w = np.hstack([np.zeros((du_dy.shape[0], 1)), du_dy]) # Pad first column with zeros

    Txy_e = nu * (du_dy_e + d_dx_forward(v, dx))
    Txy_w = nu * (du_dy_w + d_dx_backward(v, dx))
    dTxy_dx = (Txy_e - Txy_w) / dx

    v_n = avg_y_forward(v)
    v_s = avg_y_backward(v)
    dvv_dy = (v_n**2 - v_s**2) / dy

    v_e = avg_x_forward(v)
    v_w = avg_x_backward(v)

    avg_u_y = avg_y_forward(u)[:-1, :] # Chop off last row of avg_u_y
    u_e = np.hstack([avg_u_y, np.zeros((avg_u_y.shape[0], 1))]) # Pad last column with zeros to match v-shape (N+1, N+2)
    u_w = np.hstack([np.zeros((avg_u_y.shape[0], 1)), avg_u_y]) # Pad first column with zeros

    duv_dx = (u_e * v_e - u_w * v_w) / dx

    H = -duv_dx - dvv_dy + dTyy_dy + dTxy_dx

    return H

def velocity_boundary_conditions(u, v, u_lid=0.0, couvette=False):
    # Apply v=0 boundary conditions
    v[0, :] = 0             # Bottom wall
    v[-1, :] = 0            # Top wall
    if not couvette:
        v[:, 0] = -v[:, 1]      # Left wall
        v[:, -1] = -v[:, -2]    # Right wall

    # Apply u=0 boundary conditions
    u[0, :] = -u[1, :]      # Bottom wall
    u[-1, :] = 2 * u_lid - u[-2, :]  # Top wall (moving lid)
    if not couvette:
        u[:, 0] = 0             # Left wall
        u[:, -1] = 0            # Right wall    

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

    # Extract input parameters from input_file
    df = pd.read_csv(input_file)
    trial_ids = df['trial'].tolist()
    N_values = df['N'].tolist()
    dt_values = df['dt'].tolist()
    Re_values = df['Re'].tolist()
    alpha_values = df['alpha'].tolist()
    
    main()