import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os

"""
File:   channel_flow.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D steady-state channel flow of a viscous, incompressible Newtonian fluid
        using an explicit scheme with artificial compressibility. The governing equation is the 2-D
        Navier-Stokes equation.
"""

def main():
    # Initialize plot settings
    figure, axes = plt.subplots(figsize=(5, 4))
    axes.set_ylabel('y', fontsize=12)
    axes.tick_params(labelsize=12)
    axes.set_xlabel('u', fontsize=12)
    
    # Write header to output file
    with open(output_file, 'w') as file_out:
        file_out.write('trial,Nx,Ny,L,H,U0,P0,dt,tf,beta,nu,plot_x,dx,dy,Re,Ma,CFL,MAE,Time\n')
    
    for trial, Nx, Ny, L, H, u_inlet, p_inlet, dt, t_final, beta, nu, plot_x in zip(trial_ids, Nx_values, Ny_values, L_values, H_values, U_values, P_values, dt_values, tf_values, beta_values, nu_values, plot_x_values):
        # Create grid with ghost cells
        (x, y), dx, dy = create_grid(Nx, L, Ny, H, ghost_cells=True)

        # Initialize solution with ghost cells
        inv_Re = nu / u_inlet / H
        p = np.zeros((Ny, Nx + 1))
        u = u_inlet * np.ones((Ny, Nx + 1))
        v = np.zeros((Ny, Nx + 1))

        # Apply initial boundary conditions
        apply_boundary_conditions(p, u, v, p_inlet, u_inlet, 0.0)

        # Solve the governing equations using the explicit scheme
        t = 0
        start_time = time.time()
        while t < t_final:
            # Compute spatial derivatives with central differences
            dE1_dx, dE2_dx, dE3_dx = d_dx(u/beta, p+u**2, u*v, dx)
            dF1_dy, dF2_dy, dF3_dy = d_dy(v/beta, u*v, p+v**2, dy)
            d2DU1_dx2, d2DU2_dx2, d2DU3_dx2 = d2_dx2(np.zeros((Ny, Nx + 1)), u, v, dx)
            d2DU1_dy2, d2DU2_dy2, d2DU3_dy2 = d2_dy2(np.zeros((Ny, Nx + 1)), u, v, dy)

            # Compute dU and add onto solution fields
            p += dt * (-dE1_dx - dF1_dy + inv_Re * (d2DU1_dx2 + d2DU1_dy2))
            u += dt * (-dE2_dx - dF2_dy + inv_Re * (d2DU2_dx2 + d2DU2_dy2))
            v += dt * (-dE3_dx - dF3_dy + inv_Re * (d2DU3_dx2 + d2DU3_dy2))

            # Apply boundary conditions
            apply_boundary_conditions(p, u, v, p_inlet, u_inlet, 0.0)

            t += dt
        end_time = time.time()

        # Compute analytical solution and MAE
        u_analytical = 0.25 * (1 - x**2)
        mae = np.mean(np.abs(u_analytical - u))

        # Compute Ma and CFL for stability assessment
        Ma = np.sqrt(beta * (u**2 + v**2))
        max_Ma = np.max(Ma)
        CFL = dt / np.min([dx, dy]) / np.sqrt(beta)

        # Write results to output plot and file
        x_index = np.argmin(np.abs(x[0,:] - plot_x))
        axes.plot(u[:,x_index], y[:,x_index], color=f'C{trial-1}', label=f"Numerical: Trial {trial}")
        axes.plot(u[:,x_index], y[:,x_index], '--', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        with open(output_file, 'a') as file_out:
            file_out.write(f"{trial},{Nx},{Ny},{L},{H},{u_inlet},{p_inlet},{dt},{t_final},{beta},{nu},{plot_x},{x_index},{dx},{dy},{1/inv_Re},{max_Ma},{CFL},{mae},{end_time - start_time}\n")

    # Save output plot
    figure.tight_layout()
    axes.legend(loc='lower right')
    figure.savefig(plot_file)

def create_grid(Nx, L, Ny=20, H=1, ghost_cells=False):
    dx = L / Nx
    dy = H / Ny

    if ghost_cells:
        x = np.linspace(dx/2, L + dx/2, Nx + 1)
    else:
        x = np.linspace(dx/2, L - dx/2, Nx)
    y = np.linspace(dy/2, H - dy/2, Ny)

    return np.meshgrid(x, y), dx, dy

def d_dx(p, u, v, dx):
    p_padded = np.pad(p, ((0,0), (1,1)), mode='constant', constant_values=0)
    u_padded = np.pad(u, ((0,0), (1,1)), mode='constant', constant_values=0)
    v_padded = np.pad(v, ((0,0), (1,1)), mode='constant', constant_values=0)

    dp_dx = p_padded[:,2:] - p_padded[:,:-2]
    dp_dx /= 2 * dx
    du_dx = u_padded[:,2:] - u_padded[:,:-2]
    du_dx /= 2 * dx
    dv_dx = v_padded[:,2:] - v_padded[:,:-2]
    dv_dx /= 2 * dx

    return dp_dx, du_dx, dv_dx

def d_dy(p, u, v, dy):
    p_padded = np.pad(p, ((1,1), (0,0)), mode='constant', constant_values=0)
    u_padded = np.pad(u, ((1,1), (0,0)), mode='constant', constant_values=0)
    v_padded = np.pad(v, ((1,1), (0,0)), mode='constant', constant_values=0)

    dp_dy = p_padded[2:,:] - p_padded[:-2,:]
    dp_dy /= 2 * dy
    du_dy = u_padded[2:,:] - u_padded[:-2,:]
    du_dy /= 2 * dy
    dv_dy = v_padded[2:,:] - v_padded[:-2,:]
    dv_dy /= 2 * dy

    return dp_dy, du_dy, dv_dy

def d2_dx2(p, u, v, dx):
    p_padded = np.pad(p, ((0,0), (1,1)), mode='constant', constant_values=0)
    u_padded = np.pad(u, ((0,0), (1,1)), mode='constant', constant_values=0)
    v_padded = np.pad(v, ((0,0), (1,1)), mode='constant', constant_values=0)

    d2p_dx2 = p_padded[:,2:] - 2 * p_padded[:,1:-1] + p_padded[:,:-2]
    d2p_dx2 /= dx * dx
    d2u_dx2 = u_padded[:,2:] - 2 * u_padded[:,1:-1] + u_padded[:,:-2]
    d2u_dx2 /= dx * dx
    d2v_dx2 = v_padded[:,2:] - 2 * v_padded[:,1:-1] + v_padded[:,:-2]
    d2v_dx2 /= dx * dx

    return d2p_dx2, d2u_dx2, d2v_dx2

def d2_dy2(p, u, v, dy):
    p_padded = np.pad(p, ((1,1), (0,0)), mode='constant', constant_values=0)
    u_padded = np.pad(u, ((1,1), (0,0)), mode='constant', constant_values=0)
    v_padded = np.pad(v, ((1,1), (0,0)), mode='constant', constant_values=0)

    d2p_dy2 = p_padded[2:,:] - 2 * p_padded[1:-1,:] + p_padded[:-2,:]
    d2p_dy2 /= dy * dy
    d2u_dy2 = u_padded[2:,:] - 2 * u_padded[1:-1,:] + u_padded[:-2,:]
    d2u_dy2 /= dy * dy
    d2v_dy2 = v_padded[2:,:] - 2 * v_padded[1:-1,:] + v_padded[:-2,:]
    d2v_dy2 /= dy * dy

    return d2p_dy2, d2u_dy2, d2v_dy2

def apply_boundary_conditions(p, u, v, p_inlet, u_inlet, v_inlet):
    # Apply Dirichlet boundary conditions
    p[:,0] = p_inlet
    u[:,0] = u_inlet
    v[:,0] = v_inlet

    # Apply Neumann boundary conditions
    p[:,-1] = p[:,-2]
    u[:,-1] = u[:,-2]
    v[:,-1] = v[:,-2]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input .csv file with parameters and trials')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output .csv file to save results')
    parser.add_argument('-p', '--plot_file', type=str, default=None, help='Output .png file to save results')
    parser.add_argument('-f', '--base_folder', type=str, default='./', help='Base folder for trials (optional)')
    args = parser.parse_args()

    # Extract parameters from arguments
    base_folder = args.base_folder
    input_file = args.input_file
    input_file = os.path.join(base_folder, input_file)
    output_file = args.output_file
    output_file = os.path.join(base_folder, output_file)
    plot_file = args.plot_file
    plot_file = os.path.join(base_folder, plot_file)

    # Extract input parameters from input_file
    df = pd.read_csv(input_file)
    trial_ids = df['trial'].tolist()
    Nx_values = df['Nx'].tolist()
    Ny_values = df['Ny'].tolist()
    L_values = df['L'].tolist()
    H_values = df['H'].tolist()
    U_values = df['U0'].tolist()
    P_values = df['P0'].tolist()
    dt_values = df['dt'].tolist()
    tf_values = df['tf'].tolist()
    beta_values = df['beta'].tolist()
    nu_values = df['nu'].tolist()
    plot_x_values = df['plot_x'].tolist()
    
    main()