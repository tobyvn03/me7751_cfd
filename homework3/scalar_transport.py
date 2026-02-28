import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os

"""
File:   scalar_transport.py
Author: Toby Viet Nguyen
Desc:   This script solves the 1-D unsteady transport equation ∂φ/∂t + U ∂φ/∂x = ∂/∂x (Γ ∂φ/∂x)
        given constant U, Γ using an explicit, implicit, and Crank-Nicolson scheme.
"""

def main():
    figure, axes = plt.subplots(figsize=(8, 6))
    axes.set_title(plot_title, fontsize=16)
    axes.set_xlabel('x', fontsize=14)
    axes.set_ylabel('\phi', fontsize=14)
    axes.tick_params(labelsize=12)
    with open(output_file, 'w') as file_out:
        file_out.write('N, U, G, tf, ti, dt, scheme, CFL, MAE, Time\n')
    
    for trial, N, U, G, tf, ti, dt, scheme in zip(trial_ids, N_values, U_values, G_values, tf_values, ti_values, dt_values, schemes):
        # Create grid and compute source term Q and T_initial
        x, h = create_grid(N)
        phi_initial = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-10)**2)

        # Solve the Poisson equation using the specified method
        start_time = time.time()
        if scheme == 'explicit':
            phi_numerical = explicit_scheme(N, phi_initial, U, G, dt, tf-ti, h)
        elif scheme == 'implicit':
            phi_numerical = implicit_scheme(N, phi_initial, U, G, dt, tf-ti, h)
        elif scheme == 'crank-nicolson':
            phi_numerical = crank_nicolson_scheme(N, phi_initial, U, G, dt, tf-ti, h)
        end_time = time.time()

        # Compute analytical solution and MAE
        if U == 1 and G == 0.01:
            phi_analytical = (4*np.pi*G*tf)**(-0.5) * np.exp(-(x-U*tf)**2/(4*G*tf))
            mae = np.mean(np.abs(phi_analytical - phi_numerical))
        elif U == 1 and G == 0:
            phi_analytical = (0.4*np.pi)**(-0.5) * np.exp(-2.5*(x-U*tf)**2)
            mae = np.mean(np.abs(phi_analytical - phi_numerical))
    
        # Write results to output plot and file
        axes.plot(x, phi_numerical, label=f"Trial {trial}")
        with open(output_file, 'a') as file_out:
            file_out.write(f"{N}, {U}, {G}, {tf}, {ti}, {dt}, {scheme}, {U*dt/h}, {mae}, {end_time - start_time}\n")

def create_grid(N):
    h = 1 / (N - 1)
    x = np.linspace(5, 45, N)
    return x, h

def explicit_scheme(N, phi, U, G, dt, T, h):
    for _ in range(int(T/dt)):
        phi_new = np.copy(phi)
        for i in range(1, N-1):
            phi_new[i] = phi[i] - U * (phi[i] - phi[i-1]) * dt / h + G * (phi[i+1] - 2*phi[i] + phi[i-1]) * dt / h**2
        phi = phi_new
    return phi_new

def implicit_scheme(N, phi, U, G, dt, T, h):
    aW = -U * dt / h - G * dt / h**2
    aP = 1 + 2 * G * dt / h**2
    aE = G * dt / h**2
    b = phi.copy()
    for _ in range(int(T/dt)):
        for i in range(1, N-1):
            b[i] = phi[i]
        phi_new = thomas_algorithm(N, aW, aP, aE, b)
        phi = phi_new
    return phi_new

def crank_nicolson_scheme(N, phi, U, G, dt, T, h):
    aW = -0.5 * U * dt / h - 0.5 * G * dt / h**2
    aP = 1 + G * dt / h**2
    aE = 0.5 * G * dt / h**2
    b = phi.copy()
    for _ in range(int(T/dt)):
        for i in range(1, N-1):
            b[i] += 0.5 * U * (phi[i+1] - phi[i-1]) * dt / h - 0.5 * G * (phi[i+1] - 2*phi[i] + phi[i-1]) * dt / h**2
        phi_new = thomas_algorithm(N, aW, aP, aE, b)
        phi = phi_new
    return phi_new

def thomas_algorithm(N, aW, aP, aE, b):
    # Forward substitution for TDMA
    for i in range(1, N):
        aP[i] -= aW[i] * aE[i-1] / aP[i-1]
        b[i] -= aW[i] * b[i-1] / aP[i-1]
    # Backward substitution for TDMA
    phi = np.zeros(N)
    phi[-1] = b[-1] / aP[-1]
    for i in reversed(range(0, N-1)):
        phi[i] = (b[i] - aE[i] * phi[i+1]) / aP[i]
    return phi

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Poisson solver")
    parser.add_argument('-i', '--input_file', type=str, default=None, help='Input .csv file with parameters and trials')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output .csv file with results and MAE for each trial')
    parser.add_argument('-x', '--plot_x', nargs='+', type=float, help='X values for plotting')
    parser.add_argument('-t', '--plot_t', nargs='+', type=float, help='T values for plotting')
    parser.add_argument('-f', '--base_folder', type=str, default='./', help='Base folder for input, output, and plots (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output for debugging')
    args = parser.parse_args()

    # Extract parameters from arguments
    base_folder = args.base_folder
    input_file = args.input_file
    input_file = os.path.join(base_folder, input_file)
    output_file = args.output_file
    output_file = os.path.join(base_folder, output_file)

    # Use output_file as plot title and plot file
    plot_title = output_file.split('/')[-1].replace('.csv', '').split('_')[1:].replace('_', ' ').title()
    plot_file = output_file.replace('.csv', '.png')

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
    plot_t = args.plot_t
    verbose = args.verbose

    main()