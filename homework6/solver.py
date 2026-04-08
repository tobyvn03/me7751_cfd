import pandas as pd
import numpy as np
import argparse
import time
import os
# from finite_differences import *
from plotting import *

"""
File:   lid_driven_flow.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D lid-driven cavity flow of a viscous,
        incompressible Newtonian fluid using an explicit projection method.
"""

def main():
    # Initialize plot settings
    # fig_u, axes_u = initialize_plot(xlabel='u', ylabel='y')
        
    # Write header to summary output
    # with open(output_file, 'w') as file_out:
    #     file_out.write('trial,N,dt,Re,alpha,t,h,MAE,Time\n')
    
    for trial, N, dt, Re, alpha in zip(trial_ids, N_values, dt_values, Re_values, alpha_values):
        # Get coordinate grid at cell centers

        # Initialize solution with ghost cells

        # Write header to verbose output
        # if verbose:
        #     with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'w') as file_out:
        #         file_out.write('t        CFL_adv    |du/dt|    |dv/dt|    |dp/dt|    k\n')

        # Solve the governing equations using the explicit scheme
        t = 0.0
        step = 0
        start_time = time.time()
        while True:
            # Solve equation

            # Update fields and apply boundary conditions

            # Compute max time derivative for convergence check

            # Write results to verbose output
            # if verbose:
            #     speed = np.sqrt(u_new[1:-1, 1:]**2 + v_new[1:, 1:-1]**2)
            #     speed_max = np.max(speed)
            #     CFL = speed_max * dt / h
            #     with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'a') as file_out:
            #         file_out.write(f'{t:<8.4f} {CFL:<10.4e} {du_dt_max:<10.4e} {dv_dt_max:<10.4e} {dp_dt_max:<10.4e} {k:<6}\n')

            # Skip plotting if any of the time derivatives are NaN
            # if np.isnan(du_dt_max) or np.isnan(dv_dt_max):
            #     continue

            # Graph intermediate heatmaps if requested
            # if step % 500 == 0 and heatmaps:
            #     divergence = d_dx_forward(u, h)[1:-1, 1:] + d_dy_forward(v, h)[1:, 1:-1]
            #     plot_heatmap(divergence[1:,1:], title=f'{plot_file}_divergence_trial{trial}')
            #     plot_heatmap(p[1:-1, 1:-1], title=f'{plot_file}_pressure_trial{trial}')
            #     plot_heatmap(u[1:-1, 1:], title=f'{plot_file}_xvelocity_trial{trial}')
            #     plot_heatmap(v[1:, 1:-1], title=f'{plot_file}_yvelocity_trial{trial}')
            # if step % 500 == 0 and streamlines:
            #     plot_fluid_streamlines(u, v, x, y, title=f'{plot_file}_streamlines_trial{trial}')

            # Check for convergence and plot results if converged
            # tolerance = 1e-6
            # if du_dt_max < tolerance and dv_dt_max < tolerance:
            #     divergence = d_dx_forward(u, h)[1:-1, 1:] + d_dy_forward(v, h)[1:, 1:-1]
            #     plot_heatmap(divergence[1:,1:], title=f'{plot_file}_divergence_trial{trial}')
            #     plot_heatmap(p[1:-1, 1:-1], title=f'{plot_file}_pressure_trial{trial}')
            #     plot_heatmap(u[1:-1, 1:], title=f'{plot_file}_xvelocity_trial{trial}')
            #     plot_heatmap(v[1:, 1:-1], title=f'{plot_file}_yvelocity_trial{trial}')
            #     plot_fluid_streamlines(u, v, x, y, title=f'{plot_file}_streamlines_trial{trial}')
            #     break

            t += dt
            step += 1
        end_time = time.time()

        # Plot x-velocity profile
        # y_interior = y[1:-1, 0]
        # x_index = np.argmin(np.abs(x[0, :] - 1/2))
        # u_interior = u[1:-1, x_index]
        # axes_u.plot(u_interior, y_interior, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")

        # Plot analytical solution and compute MAE
        # axes_u.plot(u_analytical, y_N129[y_indices, 0], 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")

        # Write results to summary file
        # with open(output_file, 'a') as file_out:
        #     file_out.write(f"{trial},{N},{dt},{Re},{alpha},{t},{h},{MAE},{end_time - start_time}\n")

    # Save output plots
    fig_u.savefig(plot_file + '_centerline_xvelocity.png')

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