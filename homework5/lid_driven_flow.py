import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os
from finite_differences import *
from iterative_solve import sor_method

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
        pressure_boundary_conditions(p)
        velocity_boundary_conditions(u, v)

        # Write header to verbose output

        # Solve the governing equations using the explicit scheme
        t = 0
        k = 0
        start_time = time.time()
        while True:
            # Compute intermediate velocity fields G and H
            G = calculate_G(u, v, nu, h, h)
            H = calculate_H(u, v, nu, h, h)

            # Compute RHS of Poisson equation for pressure correction
            dG_dx = d_dx_backward(G, h)
            dH_dy = d_dy_backward(H, h)
            dG_dx_padded = np.hstack([dG_dx, np.zeros((dG_dx.shape[0], 1))]) # Pad the last column of dG_dx with zeros to match p-shape (N+2, N+2)
            dH_dy_padded = np.vstack([dH_dy, np.zeros((1, dH_dy.shape[1]))]) # Pad the last row of dH_dy with zeros       
            RHS = dG_dx_padded + dH_dy_padded

            # Solve Poisson equation for pressure correction using SOR method
            p, _, _ = sor_method(N + 2, h, RHS, p)

            # Update velocity fields using pressure correction
            dp_dx = d_dx_forward(p, h)[:, :-1] # Chop off last column of dp_dx
            dp_dy = d_dy_forward(p, h)[:-1, :] # Chop off last row of dp_dy
            du_dt = G - dp_dx
            dv_dt = H - dp_dy
            u += du_dt * dt
            v += dv_dt * dt
            velocity_boundary_conditions(u, v)

            # Write results to verbose output            
            du_dt_max = np.max(du_dt)
            dv_dt_max = np.max(dv_dt)
            if verbose:
                speed = np.sqrt(u**2 + v**2)
                speed_max = np.max(speed)
                CFL = speed_max * dt / h
                with open(output_file.replace('.csv', f'_trial{trial}.csv'), 'a') as file_out:
                    file_out.write(f'{k:<6} {t:<8.4f} {CFL:<10.4e} {du_dt_max:<10.4e} {dv_dt_max:<10.4e}\n')

            tolerance = 1e-12
            if du_dt_max < tolerance and dv_dt_max < tolerance:
                break

            t += dt
            k += 1
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

    du_dt = -duu_dx - duv_dy + dTxx_dx + dTxy_dy

    return du_dt

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

    dv_dt = -duv_dx - dvv_dy + dTyy_dy + dTxy_dx

    return dv_dt

def pressure_boundary_conditions(p):
    # Apply dp/dn=0 boundary condition
    p[:, 0] = p[:, 1]       # Left wall
    p[:, -1] = p[:, -2]     # Right wall
    p[0, :] = p[1, :]       # Bottom wall
    p[-1, :] = p[-2, :]     # Top wall

def velocity_boundary_conditions(u, v, u_lid=0.0):
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