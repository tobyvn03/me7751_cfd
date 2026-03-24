import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os
from finite_differences import *

"""
File:   lid_driven_flow.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D lid-driven cavity flow of a viscous,
        incompressible Newtonian fluid using an explicit projection method.
"""

def main():
    # Initialize plot settings
        
    # Write header to summary output
    
    for trial, N, dt, Re, epsilon, plot_x in zip(trial_ids, N_values, dt_values, Re_values, epsilon_values, plot_x_values):
        # Get coordinate grid at cell centers
        (x, y), h = create_grid(N, H=1, ghost_cells=True)
        nu = 1/Re

        # Initialize solution with ghost cells
        p = np.ones((N + 2, N + 2))
        u = np.zeros((N + 2, N + 1))
        v = np.zeros((N + 1, N + 2))
        apply_boundary_conditions(p, u, v)
        
        # Write header to verbose output

        # Solve the governing equations using the explicit scheme
        t = 0
        start_time = time.time()
        while True:
            u += du_dt(p, u, v, nu, h, h) * dt
            v += dv_dt(p, u, v, nu, h, h) * dt

            apply_boundary_conditions(p, u, v)

            t += dt
        end_time = time.time()

        # Write results to summary file

    # Save output plots

def initialize_plot(xlabel, ylabel, xlim=None, ylim=None):
    figure, axes = plt.subplots(figsize=(5, 4))
    axes.tick_params(labelsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    axes.set_xlabel(xlabel, fontsize=12)
    figure.tight_layout()
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    return figure, axes

def create_grid(N, H, ghost_cells=False):
    h = H / N
    if ghost_cells:
        x = np.linspace(-h/2, H + h/2, N + 2)
        y = np.linspace(-h/2, H + h/2, N + 2)
    else:
        x = np.linspace(h/2, H - h/2, N)
        y = np.linspace(h/2, H - h/2, N)
    return np.meshgrid(x, y), h

def du_dt(p, u, v, nu, dx, dy):
    dTxx_dx = 2 * nu * d2_dx2(u, dx)

    Txy_n = nu * (d_dy_forward(u, dy) + d_dx_forward(v, dx)) # Figure out subslice
    Txy_s = nu * (d_dy_backward(u, dy) + d_dx_forward(v, dx)) # Figure out subslice
    dTxy_dy = (Txy_n - Txy_s) / dy

    dp_dx = d_dx_forward(p, dx)

    u_e = avg_x_forward(u)
    u_w = avg_x_backward(u)
    duu_dx = (u_e**2 - u_w**2) / dx

    u_n = avg_y_forward(u)
    u_s = avg_y_backward(u)
    v_n = avg_x_forward(v) # Figure out subslice
    v_s = avg_x_forward(v) # Figure out subslice
    duv_dy = (u_n * v_n - u_s * v_s) / dy

    du_dt = -duu_dx - duv_dy - dp_dx + dTxx_dx + dTxy_dy

    return du_dt

def dv_dt(p, u, v, nu, dx, dy):
    dTyy_dy = 2 * nu * d2_dy2(v, dy)

    Txy_e = nu * (d_dy_forward(u, dy) + d_dx_forward(v, dx)) # Figure out subslice
    Txy_w = nu * (d_dy_forward(u, dy) + d_dx_backward(v, dx)) # Figure out subslice
    dTxy_dx = (Txy_e - Txy_w) / dx

    dp_dy = d_dy_forward(p, dy)

    v_n = avg_y_forward(v)
    v_s = avg_y_backward(v)
    dvv_dy = (v_n**2 - v_s**2) / dy

    u_e = avg_x_forward(u) # Figure out subslice
    u_w = avg_x_backward(u) # Figure out subslice
    v_e = avg_x_forward(v) # Figure out subslice
    v_w = avg_x_backward(v) # Figure out subslice
    duv_dx = (u_e * v_e - u_w * v_w) / dx

    dv_dt = -duv_dx - dvv_dy - dp_dy + dTyy_dy + dTxy_dx

    return dv_dt

def apply_boundary_conditions(p, u, v, u_lid=0.0):
    # Apply dp/dn=0 boundary condition
    p[:, 0] = p[:, 1]       # Left wall
    p[:, -1] = p[:, -2]     # Right wall
    p[0, :] = p[1, :]       # Bottom wall
    p[-1, :] = p[-2, :]     # Top wall

    # Apply v=0 boundary conditions
    v[:, 0] = -v[:, 1]      # Left wall
    v[:, -1] = -v[:, -2]    # Right wall
    v[0, :] = 0             # Bottom wall
    v[-1, :] = 0            # Top wall

    # Apply u=0 boundary conditions
    u[:, 0] = 0             # Left wall
    u[:, -1] = 0            # Right wall
    u[0, :] = -u[1, :]      # Bottom wall
    
    # u=u_lid on the top wall only
    u[-1, :] = 2 * u_lid - u[-2, :]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input .csv file with parameters and trials')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output .csv file to save results')
    parser.add_argument('-p', '--plot_file', type=str, default=None, help='Output .png file to save results')
    parser.add_argument('-f', '--base_folder', type=str, default='./', help='Base folder for trials (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Write verbose output file')
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

    # Extract input parameters from input_file
    df = pd.read_csv(input_file)
    trial_ids = df['trial'].tolist()
    N_values = df['N'].tolist()
    dt_values = df['dt'].tolist()
    a_values = df['a'].tolist()
    Re_values = df['Re'].tolist()
    epsilon_values = df['epsilon'].tolist()
    plot_x_values = df['plot_x'].tolist()
    
    main()