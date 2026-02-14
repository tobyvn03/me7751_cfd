import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

"""
File:   scalar_transport.py
Author: Toby Viet Nguyen
Desc:   This script solves the 1-D steady transport equation U ∂φ/∂x = ∂/∂x (Γ ∂φ/∂x) + Q
        given constant U and non-constant Γ, Q, using central differences and TDMA.
"""

def homework1b():
    # Input parameters
    U = 1.0
    Gamma = 0.1
    Q = 0.0
    output_filename = 'results/homework1b_plot.png'

    # Complete homework 1b for N up to 1000
    for N in np.linspace(10, 1000, 100, dtype=int):
        # Discretize domain
        x = np.linspace(0, 1, N)
        dx = 1 / (N - 1)

        # Solve for φ    
        start_time = time.time()
        phi_numerical = solve_solution(N, dx, U, Gamma * np.ones(N-1), Q * np.ones(N), verbose=False)
        end_time = time.time()

        # Compute analytical solution and MAE
        phi_actual = (1 - np.exp(U*x/Gamma)) / (np.exp(U/Gamma) - 1) + 1
        mae = np.mean(np.abs(phi_actual - phi_numerical))
        print(f"{N}, {mae}, {end_time - start_time}")

        # Plot numerical solution for N = 10, 20, 30, 40, 50
        if N in [10, 20, 30, 40, 50]:
            plt.plot(x, phi_numerical, label=f'Numerical for N = {N}')
        if N == 1000:
            plt.plot(x, phi_actual, label='Analytical', linestyle='--')

    # Plotting
    plt.xlabel('x')
    plt.ylabel('φ')
    plt.title('1-D Steady Transport Solution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(output_filename)

def homework1c():
    # Input parameters
    N = 20
    U = 0.0
    Gamma0 = 0.1
    Gamma1 = 0.1
    output_filename = 'results/homework1c_plot.png'

    # Discretize domain
    x = np.linspace(0, 1, N)
    dx = 1 / (N - 1)
    phi_old = 1 - x

    # Complete homework 1c for Q(x) = 0, Q(x) = 0.1, and Q(x) = 0.1*x
    for Q0, Q1 in [(0.0, 0.0), (0.0, 0.1), (0.1, 0.0)]:
        # Compute Q(x) for current case
        Q = Q1 * x + Q0

        # Solve for φ using iterative method
        start_time = time.time()
        phi_numerical, k = solve_iterative_solution(N, dx, U, Gamma0, Gamma1, Q, phi_old, verbose=False)
        end_time = time.time()

        # Compute analytical solution and MAE
        if Q1 == 0 and Q0 == 0:
            phi_actual = -1 + np.sqrt(4 - 3*x)
        elif Q1 == 0 and Q0 == 0.1:
            phi_actual = -1 + np.sqrt(4 - 2*x - x**2)
        elif Q1 == 0.1 and Q0 == 0:
            phi_actual = -1 + np.sqrt(4 - (8/3)*x - x**3/3)
        mae = np.mean(np.abs(phi_actual - phi_numerical))
        print(f"{Q1}, {Q0}, {mae}, {end_time - start_time}, {k}")
        plt.plot(x, phi_numerical, label=f'Numerical for Q(x) = {Q1}*x + {Q0}')
        plt.plot(x, phi_actual, label=f'Analytical for Q(x) = {Q1}*x + {Q0}', linestyle='--')
    
    # Plotting
    plt.xlabel('x')
    plt.ylabel('φ')
    plt.title('1-D Steady Transport Solution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(output_filename)

def solve_iterative_solution(N, dx, U, Gamma0, Gamma1, Q, initial_guess, tol=1e-6, max_iter=1000, verbose=False):
    # Initialize φ with linear guess
    phi_old = initial_guess.copy()

    # Iteratively solve for φ
    k = 0
    while k < max_iter:
        # Calculate Γ at i±1/2 using current φ
        Gamma = Gamma0 + Gamma1 * (phi_old[1:] + phi_old[:-1]) / 2  # Average Γ at i±1/2
        phi_new = solve_solution(N, dx, U, Gamma, Q, verbose=False)
        if np.max(np.abs(phi_new - phi_old)) < tol:
            break
        phi_old = phi_new
        k += 1
    if verbose:
        print(f"Converged in {k} iterations")
    return phi_new, k

def solve_solution(N, dx, U, Gamma, Q, verbose=False):
    # Compute Γ at faces
    GammaW = np.insert(Gamma, 0, 0.0)  # Γ at west faces
    GammaE = np.append(Gamma, 0.0)   # Γ at east faces
    
    # Initialize coefficient band vectors and RHS vector
    aW = GammaW/dx**2 + U/(2*dx)
    aW[0] = 0.0  # No west neighbor for first node
    aP = -(GammaW + GammaE)/dx**2
    aE = GammaE/dx**2 - U/(2*dx)
    aE[-1] = 0.0  # No east neighbor for last node
    b = -Q
    if verbose:
        print("Before Boundary Conditions")
        print(f"GammaW{GammaW} = \n{GammaW}")
        print(f"GammaE{GammaE} = \n{GammaE}")
        print(f"aW{aW.shape} = \n{aW}")
        print(f"aP{aP.shape} = \n{aP}")
        print(f"aE{aE.shape} = \n{aE}")
        print(f"b{b.shape} = \n{b}")

    # Apply boundary conditions
    b[0] = 1.0  # φ(0) = 1
    b[-1] = 0.0  # φ(1) = 0
    aW[-1] = 0.0
    aE[0] = 0.0
    aP[0] = 1.0
    aP[-1] = 1.0
    if verbose:
        print("After Boundary Conditions")
        print(f"aW{aW.shape} = \n{aW}")
        print(f"aP{aP.shape} = \n{aP}")
        print(f"aE{aE.shape} = \n{aE}")
        print(f"b{b.shape} = \n{b}")

    # Forward substitution for TDMA
    for i in range(1, N):
        aP[i] -= aW[i] * aE[i-1] / aP[i-1]
        b[i] -= aW[i] * b[i-1] / aP[i-1]
    if verbose:
        print("After Forward Substitution")
        print(f"aP{aP.shape} = \n{aP}")
        print(f"b{b.shape} = \n{b}")
    
    # Backward substitution for TDMA
    phi = np.zeros(N)
    phi[-1] = b[-1] / aP[-1]
    for i in reversed(range(0, N-1)):
        phi[i] = (b[i] - aE[i] * phi[i+1]) / aP[i]
    if verbose:
        print("After Backward Substitution")
        print(f"phi{phi.shape} = \n{phi}")

    return phi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve 1-D steady transport equation with TDMA")
    parser.add_argument("-b", "--homework1b", action="store_true", help="Run homework 1b with hard-coded parameters")
    parser.add_argument("-c", "--homework1c", action="store_true", help="Run homework 1c with hard-coded parameters")
    args = parser.parse_args()
    
    if args.homework1b:
        homework1b()
    if args.homework1c:
        homework1c()