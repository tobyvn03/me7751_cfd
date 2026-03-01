import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os

"""
File:   scalar_transport.py
Author: Toby Viet Nguyen
Desc:   This script solves the 1-D unsteady transport equation ∂ϕ/∂t + U ∂ϕ/∂x = ∂/∂x (Γ ∂ϕ/∂x)
        given constant U, Γ using an explicit, implicit, and Crank-Nicolson scheme.
"""

def main():
    # Set up plot
    figure, axes = plt.subplots(figsize=(5, 4))
    axes.set_title(plot_title, fontsize=16)
    axes.set_ylabel('ϕ', fontsize=12)
    axes.tick_params(labelsize=12)
    if plot_x is None:
        axes.set_xlabel('x', fontsize=12)
    if plot_x is not None:
        axes.set_xlabel('t', fontsize=12)

    # Write header to output file
    with open(output_file, 'w') as file_out:
        file_out.write('trial,N,U,G,tf,ti,dt,scheme,h,CFL,Diffusion,MAE,Time\n')
    
    for trial, N, U, G, tf, ti, dt, scheme in zip(trial_ids, N_values, U_values, G_values, tf_values, ti_values, dt_values, schemes):
        # Create grid and compute initial solution
        x, h = create_grid(N)
        phi_initial = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-10)**2)

        # Find index for plotting if plot_x is specified
        x_index = None
        if plot_x is not None:
            x_index = np.argmin(np.abs(x - plot_x))

        # Solve the transport equation using the specified scheme
        start_time = time.time()
        if scheme == 'explicit':
            phi_numerical, c, d, t_series, phi_series = explicit_scheme(phi_initial, N, U, G, tf, ti, dt, h, x_index=x_index)
        elif scheme == 'implicit':
            phi_numerical, c, d, t_series, phi_series = implicit_scheme(phi_initial, N, U, G, tf, ti, dt, h, x_index=x_index)
        elif scheme == 'crank-nicolson':
            phi_numerical, c, d, t_series, phi_series = crank_nicolson_scheme(phi_initial, N, U, G, tf, ti, dt, h, x_index=x_index)
        end_time = time.time()

        # Compute analytical solution and MAE w.r.t. x
        if plot_x is None:
            if U == 1 and G == 0.01:
                phi_analytical = (4*np.pi*G*tf)**(-0.5) * np.exp(-(x-U*tf)**2/(4*G*tf))
                mae = np.mean(np.abs(phi_analytical - phi_numerical))
            elif U == 1 and G == 0:
                phi_analytical = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-U*tf)**2)
                mae = np.mean(np.abs(phi_analytical - phi_numerical))

        # Compute analytical solution and MAE w.r.t. t if plot_x is specified
        if plot_x is not None:
            if U == 1 and G == 0.01:
                phi_analytical = (4*np.pi*G*t_series)**(-0.5) * np.exp(-(plot_x-U*np.array(t_series))**2/(4*G*np.array(t_series)))
                mae = np.mean(np.abs(phi_analytical - np.array(phi_series)))
            elif U == 1 and G == 0:
                phi_analytical = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(plot_x-U*np.array(t_series))**2)
                mae = np.mean(np.abs(phi_analytical - np.array(phi_series)))
    
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

def create_grid(N):
    h = 40 / (N - 1)
    x = np.linspace(5, 45, N)
    return x, h

def explicit_scheme(phi_initial, N, U, G, tf, ti, dt, h, x_index=None):
    c = U * dt / h
    d = G * dt / h**2
    if c > 1 or d > 0.5:
        print(f"Warning: Stability condition violated for explicit scheme (CFL={c:.2f}, Diffusion number={d:.2f})")
    
    phi = phi_initial.copy()
    phi_new = phi_initial.copy()
    apply_boundary_conditions(phi)

    t_series = np.array([])
    phi_series = np.array([])
    for t_step in range(int((tf-ti)/dt)):
        for i in range(1, N-1):
            phi_new[i] = phi[i] - c/2 * (phi[i+1] - phi[i-1]) + d * (phi[i+1] - 2*phi[i] + phi[i-1])
        apply_boundary_conditions(phi_new)
        if x_index is not None:
            t_series = np.append(t_series, ti + (t_step+1)*dt)
            phi_series = np.append(phi_series, phi_new[x_index])
        phi[:] = phi_new[:]
    return phi, c, d, t_series, phi_series

def implicit_scheme(phi_initial, N, U, G, tf, ti, dt, h, x_index=None):
    c = U * dt / h
    d = G * dt / h**2
    aW = (-c/2 - d) * np.ones(N)
    aP = (1 + 2*d) * np.ones(N)
    aE = (c/2 - d) * np.ones(N)

    phi = phi_initial.copy()
    phi_new = phi_initial.copy()
    apply_boundary_conditions(phi)
    apply_boundary_conditions(aW)
    apply_boundary_conditions(aP, value=1.0)
    apply_boundary_conditions(aE)

    # Forward elimination of matrix coefficients for TDMA
    for i in range(1, N):
        aP[i] -= aW[i] * aE[i-1] / aP[i-1]

    t_series = np.array([])
    phi_series = np.array([])
    for t_step in range(int((tf-ti)/dt)):
        phi_new = thomas_algorithm(N, aW, aP, aE, phi)
        apply_boundary_conditions(phi_new)
        if x_index is not None:
            t_series = np.append(t_series, ti + (t_step+1)*dt)
            phi_series = np.append(phi_series, phi_new[x_index])
        phi[:] = phi_new[:]
    return phi_new, c, d, t_series, phi_series

def crank_nicolson_scheme(phi_initial, N, U, G, tf, ti, dt, h, x_index=None):
    c = U * dt / h
    d = G * dt / h**2
    aW = (-c/4 - d/2) * np.ones(N)
    aP = (1 + d) * np.ones(N)
    aE = (c/4 - d/2) * np.ones(N)

    phi = phi_initial.copy()
    psi = phi_initial.copy()  # RHS vector for Crank-Nicolson
    phi_new = phi_initial.copy()
    apply_boundary_conditions(phi)
    apply_boundary_conditions(aW)
    apply_boundary_conditions(aP, value=1.0)
    apply_boundary_conditions(aE)

    # Forward elimination of matrix coefficients for TDMA
    for i in range(1, N):
        aP[i] -= aW[i] * aE[i-1] / aP[i-1]

    t_series = np.array([])
    phi_series = np.array([])
    for t_step in range(int((tf-ti)/dt)):
        for i in range(1, N-1):
            psi[i] = (c/4 + d/2) * phi[i-1] + (1 - d) * phi[i] + (-c/4 + d/2) * phi[i+1]
        phi_new = thomas_algorithm(N, aW, aP, aE, psi)
        apply_boundary_conditions(phi_new)
        if x_index is not None:
            t_series = np.append(t_series, ti + (t_step+1)*dt)
            phi_series = np.append(phi_series, phi_new[x_index])
        phi[:] = phi_new[:]
    return phi_new, c, d, t_series, phi_series

def thomas_algorithm(N, aW, aP, aE, b):
    # Forward substitution for TDMA
    for i in range(1, N):
        b[i] -= aW[i] * b[i-1] / aP[i-1]
    
    # Backward substitution for TDMA
    phi = np.zeros(N)
    phi[-1] = b[-1] / aP[-1]
    for i in reversed(range(0, N-1)):
        phi[i] = (b[i] - aE[i] * phi[i+1]) / aP[i]
    return phi

def apply_boundary_conditions(phi, value=0.0):
    phi[0] = value
    phi[-1] = value

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