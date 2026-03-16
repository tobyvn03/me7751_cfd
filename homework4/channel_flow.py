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
    # Initialize plot settings for velocity
    figure_velocity, axes_velocity = plt.subplots(figsize=(5, 4))
    axes_velocity.tick_params(labelsize=12)
    axes_velocity.set_ylabel('y', fontsize=12)
    axes_velocity.set_xlabel('u', fontsize=12)
    axes_velocity.set_ylim(0, 1)
    # axes_velocity.set_xlim(0, 1.7)

    # Initialize plot settings for pressure
    figure_pressure, axes_pressure = plt.subplots(figsize=(5, 4))
    axes_pressure.tick_params(labelsize=12)   
    axes_pressure.set_ylabel('p', fontsize=12)
    axes_pressure.set_xlabel('x', fontsize=12)
    # axes_pressure.set_ylim(0, 1)
    # axes_pressure.set_xlim(0, L_values[0])
    
    # Write header to output file
    with open(output_file, 'w') as file_out:
        file_out.write('trial,Nx,Ny,L,dt,a,Re,plot_x,k,dx,dy,CFL,MAE,Time\n')
    
    for trial, Nx, Ny, L, dt, a, Re, plot_x in zip(trial_ids, Nx_values, Ny_values, L_values, dt_values, a_values, Re_values, plot_x_values):
        # Create grid with ghost cells
        (x, y), dx, dy = create_grid(Nx, L, Ny, H=1)

        # Initialize solution with ghost cells
        p = 1 - x/L
        u = np.zeros((Ny + 2, Nx + 1))
        v = np.zeros((Ny + 2, Nx + 1))

        # Apply initial boundary conditions
        apply_boundary_conditions(p, u, v)

        # Solve the governing equations using the explicit scheme
        t = 0
        k = 0
        dt_current = dt
        start_time = time.time()
        while True:
            # Compute spatial derivatives
            dE1_dx = d_dx(a**2 * u, dx)
            dE2_dx = d_dx(p + u**2, dx)
            dE3_dx = d_dx(u * v, dx)
            dF1_dy = d_dy(a**2 * v, dy)
            dF2_dy = d_dy(u * v, dy)
            dF3_dy = d_dy(p + v**2, dy)
            d2u_dx2 = d2_dx2(u, dx)
            d2u_dy2 = d2_dy2(u, dy)
            d2v_dx2 = d2_dx2(v, dx)
            d2v_dy2 = d2_dy2(v, dy)

            # Compute CFL-based adaptive dt
            max_vel = np.max(np.abs(u)) # max(np.max(np.abs(u)), np.max(np.abs(v)))
            CFL_ac = a * dt_current / np.min([dx, dy])
            CFL_adv = max_vel * dt_current / np.min([dx, dy])
            CFL_max = max(CFL_ac, CFL_adv)
            if not np.isfinite(CFL_max):
                print(f"Run #{trial}: CFL_max became non-finite (u or p diverged). Stopping.")
                break

            # Limit dt to avoid runaway shrinking
            dt_min = 1e-12
            if CFL_max > 0.5:
                dt_current *= 0.5 / CFL_max
            elif CFL_max < 0.25:
                dt_current = min(dt, dt_current * 1.1)
            if dt_current < dt_min:
                print(f"Run #{trial}: dt dropped below {dt_min:.1e}; stopping to prevent underflow.")
                break

            # Compute dp, du, dv
            dp = dt_current * (-dE1_dx - dF1_dy)
            du = dt_current * (-dE2_dx - dF2_dy + (d2u_dx2 + d2u_dy2) / Re)
            dv = dt_current * (-dE3_dx - dF3_dy + (d2v_dx2 + d2v_dy2) / Re)

            # Update solution fields
            p += dp
            u += du
            v += dv

            # Stop if solution diverged
            if not np.isfinite(u).all() or not np.isfinite(p).all():
                print(f"Run #{trial}: solution diverged (NaN/Inf) at k={k}. Stopping.")
                break

            # Apply boundary conditions
            apply_boundary_conditions(p, u, v)

            # Compute stability metrics
            dp_dt = np.mean(np.abs(dp)) / dt_current
            du_dt = np.mean(np.abs(du)) / dt_current
            Ma = np.max(np.abs(u)) / a
            max_abs_div = np.max(np.abs(dE1_dx + dF1_dy)) / a**2
            print(f'Run #{trial}: k = {k}, t = {t:.4f}, dt = {dt_current:.4e}, CFL_ac = {CFL_ac:.4e}, CFL_adv = {CFL_adv:.4e}, Ma = {Ma:.4e}, |dp/dt| = {dp_dt:.4e}, |du/dt| = {du_dt:.4e}')

            # Break if all residuals are under tolerance, otherwise continue
            tolerance = 1e-4
            if ((dp_dt < tolerance) and (du_dt < tolerance)) or (k > 1e2):
                break

            t += dt_current
            k += 1
        end_time = time.time()

        # Compute analytical solution and MAE (only on the physical interior cells)
        y_phys = y[1:-1, 0]
        x_index = np.argmin(np.abs(x[0, :] - plot_x))
        u_phys = u[1:-1, x_index]
        u_analytical = 6 * y_phys * (1 - y_phys)
        mae = np.mean(np.abs(u_analytical - u_phys))

        # Plot results for pressure
        y_index = np.argmin(np.abs(y[:,0] - 1/2))
        axes_pressure.plot(x[y_index,:], p[y_index,:], color=f'C{trial-1}', label=f"Numerical: Trial {trial}")
        # axes_pressure.plot(p_inlet * (1 - x[0,:]/L), y[:,x_index], '--', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
        
        # Plot results for velocity (physical interior only)
        axes_velocity.plot(u_phys, y_phys, color=f'C{trial-1}', label=f"Numerical: Trial {trial}")
        axes_velocity.plot(u_analytical, y_phys, '--', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")

        # Write results to output file
        CFL = max(CFL_ac, CFL_adv)
        with open(output_file, 'a') as file_out:
            file_out.write(f"{trial},{Nx},{Ny},{L},{dt},{a},{Re},{plot_x},{k},{dx},{dy},{CFL},{mae},{end_time - start_time}\n")

    # Save output plots
    figure_velocity.tight_layout()
    figure_pressure.tight_layout()
    # axes_velocity.legend(loc='lower right')
    # axes_pressure.legend(loc='lower right')
    figure_velocity.savefig(plot_file + '_velocity.png')
    figure_pressure.savefig(plot_file + '_pressure.png')

def create_grid(Nx, L, Ny=20, H=1, ghost_cells=True):
    dx = L / Nx
    dy = H / Ny

    if ghost_cells:
        x = np.linspace(dx/2, L + dx/2, Nx + 1)
        y = np.linspace(-dy/2, H + dy/2, Ny + 2)
    else:
        x = np.linspace(dx/2, L - dx/2, Nx)
        y = np.linspace(dy/2, H - dy/2, Ny)

    return np.meshgrid(x, y), dx, dy

def d_dx(f, dx):
    df_dx = np.zeros_like(f)
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)

    return df_dx

def d_dx_upwind(f, u, dx):
    df_dx = np.zeros_like(f)

    u_center = u[:, 1:-1]
    f_center = f[:, 1:-1]
    f_left = f[:, :-2]
    f_right = f[:, 2:]

    df_dx[:, 1:-1] = np.where(
        u_center >= 0,
        (f_center - f_left) / dx,
        (f_right - f_center) / dx,
    )

    return df_dx

def d_dy(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)

    return df_dy

def d_dy_upwind(f, v, dy):
    df_dy = np.zeros_like(f)

    v_center = v[1:-1, :]
    f_center = f[1:-1, :]
    f_down = f[:-2, :]
    f_up = f[2:, :]

    df_dy[1:-1, :] = np.where(
        v_center >= 0,
        (f_center - f_down) / dy,
        (f_up - f_center) / dy,
    )

    return df_dy

def d2_dx2(f, dx):
    d2f_dx2 = np.zeros_like(f)
    d2f_dx2[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (dx * dx)

    return d2f_dx2

def d2_dy2(f, dy):
    d2f_dy2 = np.zeros_like(f)
    d2f_dy2[1:-1, :] = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / (dy * dy)

    return d2f_dy2

def apply_boundary_conditions(p, u, v, p_inlet=1.0, p_outlet=0.0, u_inlet=1.0, v_inlet=0.0):
    # Apply pressure boundary conditions (constant pressure drop)
    # p[:, 0] = p_inlet
    p[:, -1] = p[:, -2]

    # Apply velocity boundary conditions
    # Use Neumann (zero-gradient) at inlet/outlet to avoid forcing divergence
    u[:, 0] = u_inlet
    u[:, -1] = u[:, -2]

    # v is zero at inlet/outlet (no cross-flow at boundaries)
    v[:, 0] = v_inlet
    v[:, -1] = v[:, -2]

    # Apply no-slip wall boundary conditions for both velocity components
    p[0, :] = p[1, :]
    u[0, :] = -u[1, :]
    v[0, :] = -v[1, :]
    p[-1, :] = p[-2, :]
    u[-1, :] = -u[-2, :]
    v[-1, :] = -v[-2, :]

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
    dt_values = df['dt'].tolist()
    a_values = df['a'].tolist()
    Re_values = df['Re'].tolist()
    plot_x_values = df['plot_x'].tolist()
    
    main()