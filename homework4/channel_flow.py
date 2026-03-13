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
    # Write header to output file
    with open(output_file, 'w') as file_out:
        file_out.write('trial,Nx,Ny,dx,dy,U0,dt,tf,beta,Ma,Re,MAE,Time\n')
    
    for trial, N, U, G, tf, ti, dt, scheme in zip(trial_ids, N_values, U_values, G_values, tf_values, ti_values, dt_values, schemes):
        # Create grid and initialize solution
        (x, y), dx, dy = create_grid(Nx, L, Ny, H, ghost_cells=True)
        p = np.zeros((Nx, Ny))
        u = u_inlet * np.ones((Nx, Ny))
        v = np.zeros((Nx, Ny))
        inv_Re = mu / rho / u_inlet / H

        # Solve the governing equations using the explicit scheme
        start_time = time.time()
        while t < t_final:
            # Compute spatial derivatives with central differences
            dE1_dx, dE2_dx, dE3_dx = d_dx(u/beta, p+u**2, u*v, dx)
            dF1_dy, dF2_dy, dF3_dy = d_dy(v/beta, u*v, p+v**2, dy)
            d2DU1_dx2, d2DU2_dx2, d2DU3_dx2 = d2_dx2(0, u, v, dx)
            d2DU1_dy2, d2DU2_dy2, d2DU3_dy2 = d2_dx2(0, u, v, dy)

            # Compute dU and add onto solution fields
            p += dt * (-dE1_dx - dF1_dy + inv_Re * (d2DU1_dx2 + d2DU1_dy2))
            u += dt * (-dE2_dx - dF2_dy + inv_Re * (d2DU2_dx2 + d2DU2_dy2))
            v += dt * (-dE3_dx - dF3_dy + inv_Re * (d2DU3_dx2 + d2DU3_dy2))

            t += dt
        end_time = time.time()

        # Compute analytical solution and MAE w.r.t. x
        if plot_x is None:
            if U == 1 and G == 0.01:
                phi_analytical = (4*np.pi*G*tf)**(-0.5) * np.exp(-(x-U*tf)**2/(4*G*tf))
                mae = np.mean(np.abs(phi_analytical - phi_numerical))
            elif U == 1 and G == 0:
                phi_analytical = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-U*tf)**2)
                mae = np.mean(np.abs(phi_analytical - phi_numerical))
    
        # Write results to output plot and file
        if plot_x is None:
            axes.plot(x, phi_numerical, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")
            axes.plot(x, phi_analytical, '--', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        if plot_x is not None:
            axes.plot(t_series, phi_series, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")
            axes.plot(t_series, phi_analytical, '--', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        with open(output_file, 'a') as file_out:
            file_out.write(f"{trial},{N},{U},{G},{tf},{ti},{dt},{scheme},{h},{c},{d},{mae},{end_time - start_time}\n")

    figure.tight_layout()
    figure.savefig(plot_file)

def create_grid(Nx, L, Ny=21, H=1, ghost_cells=False):
    dx = L / Nx
    dy = H / Ny

    if ghost_cells:
        x = np.linspace(0, L + dx, Nx + 1)
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
    p[:,0] = p_inlet
    u[:,0] = u_inlet
    v[:,0] = v_inlet

    p[:,-1] = p[:,-2]
    u[:,-1] = u[:,-2]
    v[:,-1] = v[:,-2]
    
    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input .csv file with parameters and trials')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output .csv file to save results')
    parser.add_argument('-x', '--plot_x', type=float, default=None, help='X values for plotting')
    parser.add_argument('-f', '--base_folder', type=str, default='./', help='Base folder for trials (optional)')
    args = parser.parse_args()

    # Extract parameters from arguments
    base_folder = args.base_folder
    input_file = args.input_file
    input_file = os.path.join(base_folder, input_file)
    output_file = args.output_file
    output_file = os.path.join(base_folder, output_file)

    # Use output_file as plot title and plot file
    plot_title = output_file.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
    plot_file = os.path.join(base_folder, 'plots', output_file.split('/')[-1].replace('.csv', '.png'))

    df = pd.read_csv(input_file)
    trial_ids = df['trial'].tolist()
    N_values = df['N'].tolist()
    U_values = df['U'].tolist()
    G_values = df['G'].tolist()
    tf_values = df['tf'].tolist()
    ti_values = df['ti'].tolist()
    dt_values = df['dt'].tolist()
    schemes = df['scheme'].tolist()

    plot_x = args.plot_x
    
    main()