import numpy as np
from scipy import sparse
import time
import argparse
from plotting import plot_solution plot_field

"""
Name:   solver.py
Author: Toby Viet Nguyen
Desc:   This script solves the 2-D Burgers' equation using FEM.
"""

# Reference gradients ∇_{ξ,η} [N1, N2, N3] are constant for linear triangular elements
GRAD_REF = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

def main():
    nu = 1.0 / Re # Velocity and length scale are both 1

    # Load mesh data
    coords, elems = load_mesh(N, mesh_directory)
    # visualize_mesh(N, coords, elems, output_file=f'{mesh_directory}/mesh_{N}.png')
    row_indices = np.repeat(elems, 3, axis=1).reshape(-1)
    col_indices = np.tile(elems, (1, 3)).reshape(-1)
    num_nodes = coords.shape[0]

    # Identify boundary nodes for Dirichlet BCs (u=0 on left/right, v=0 on top/bottom)
    bc_indices_x = np.where((coords[:, 0] == 0) | (coords[:, 0] == 1))[0]
    bc_indices_y = np.where((coords[:, 1] == 0) | (coords[:, 1] == 1))[0]

    # Compute the Jacobian determinants and their inverses for all elements
    det_J, inv_J_T = compute_jacobians(coords, elems)
    areas = 0.5 * np.abs(det_J)

    # Assemble the global mass and stiffness matrices
    M = assemble_mass_matrix(areas, row_indices, col_indices, num_nodes, variation='consistent').tocsr()
    K = assemble_stiffness_matrix(areas, inv_J_T, nu, row_indices, col_indices, num_nodes).tocsr()

    # Initial condition
    U_old, V_old = initial_condition(coords)
    U_old[bc_indices_x] = 0.0
    V_old[bc_indices_y] = 0.0

    # Get average element size
    edge_lengths = np.linalg.norm(coords[elems[:, 1]] - coords[elems[:, 0]], axis=1)
    edge_lengths = np.concatenate([edge_lengths, np.linalg.norm(coords[elems[:, 2]] - coords[elems[:, 1]], axis=1)])
    edge_lengths = np.concatenate([edge_lengths, np.linalg.norm(coords[elems[:, 0]] - coords[elems[:, 2]], axis=1)])
    h = edge_lengths.mean()
    print(f"h = {h:.2e}")

    # Stability checks for explicit time-stepping
    D = nu * dt / h**2
    if D > 0.25:
        print(f"D = {D:.2e} !!!")
    else:
        print(f"D = {D:.2e}")
    U_max = np.max(np.abs(U_old))
    V_max = np.max(np.abs(V_old))
    CFL = max(U_max, V_max) * dt / h
    if CFL > 1.0:
        print(f"CFL(t=0) = {CFL:.2e} !!!")
    else:
        print(f"CFL(t=0) = {CFL:.2e}")

    # Semi-implicit time-stepping loop
    start_time = time.time()
    for t in range(Nt):
        # Compute the velocity-dependent convection matrix
        C = assemble_convection_matrix(U_old, V_old, areas, inv_J_T, elems, row_indices, col_indices, num_nodes)
        S = assemble_streamline_diffusion_matrix(U_old, V_old, areas, inv_J_T, elems, row_indices, col_indices, num_nodes, nu)

        # Freeze the advecting velocity at the previous time step and solve the transported field implicitly.
        A = (M / dt) + K + C + S
        A = A.tocsr()
        b_U = (M / dt) @ U_old
        b_V = (M / dt) @ V_old

        A_U = apply_dirichlet(A, bc_indices_x)
        A_V = apply_dirichlet(A, bc_indices_y)
        
        U_new = sparse.linalg.spsolve(A_U, b_U)
        V_new = sparse.linalg.spsolve(A_V, b_V)

        U_new[bc_indices_x] = 0.0
        V_new[bc_indices_y] = 0.0

        # Move to next time step
        U_old = U_new
        V_old = V_new

        # Check max velocity
        U_max = np.max(np.abs(U_new))
        V_max = np.max(np.abs(V_new))
        print(f"t={(t+1) * dt:.4f}, max|U|={U_max:.4e}, max|V|={V_max:.4e}")
        if np.isnan(U_max) or np.isnan(V_max):
            break

        # Save solution every plot_every time steps
        if (t + 1) % plot_every == 0:
            dump_solution(coords, U_new, V_new, output_file=f'{output_directory}/solution_{(t+1) * dt:.4f}.txt')
            plot_solution(coords, U_new, V_new, output_file=f'{output_directory}/velocity_{(t+1) * dt:.4f}.png')
            plot_field(coords, U_new, output_file=f'{output_directory}/U_{(t+1) * dt:.4f}.png', title='U Velocity')
    end_time = time.time()

    print(f"Total simulation time: {end_time - start_time:.2f} seconds")

def load_mesh(N, mesh_directory='mesh'):
    coords_file = f'{mesh_directory}/coordinates_{N}.input'
    elems_file = f'{mesh_directory}/elements_{N}.input'

    coords = np.loadtxt(coords_file, delimiter=',', dtype=float)
    elems = np.loadtxt(elems_file, delimiter=',', dtype=int) - 1 # Convert to zero-based indexing
    return coords, elems

def visualize_mesh(N, coords, elems, output_file=None):
    plt.figure(figsize=(8, 8))
    for element in elems:
        x = coords[element, 0]
        y = coords[element, 1]
        plt.fill(x, y, edgecolor='k', fill=False)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Mesh Visualization for N={N}')
    plt.axis('equal')
    if output_file:
        plt.savefig(output_file)

def compute_jacobians(coords, elems):
    det_J = np.zeros(elems.shape[0], dtype=np.float64)
    inv_J_T = np.zeros((elems.shape[0], 2, 2), dtype=np.float64)

    for i, element in enumerate(elems):
        x1, y1 = coords[element[0]]
        x2, y2 = coords[element[1]]
        x3, y3 = coords[element[2]]
        det_J[i] = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        J = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]], dtype=np.float64)
        inv_J_T[i] = np.linalg.inv(J).T
    
    return det_J, inv_J_T

def assemble_mass_matrix(areas, row_indices, col_indices, num_nodes, variation='consistent'):
    # The Jacobian J transforms the reference triangle to the actual triangle.
    # |J| is twice the area of the triangle, so Area = 0.5 * |J|.
    if variation == 'consistent':
        template = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
        Me_local = (areas[:, None, None] / 12.0) * template
    elif variation == 'lumped':
        Me_local = (areas[:, None, None] / 3.0) * np.eye(3, dtype=np.float64)
    data = Me_local.reshape(-1)

    M = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    return M

def compute_physical_shape_gradients(inv_J_T):
    # GRAD_REF stores [dN/dxi, dN/deta] as row vectors, so the physical gradients
    # are row-wise products with J^{-1}. The code stores J^{-T}, hence the transpose.
    inv_J = np.transpose(inv_J_T, (0, 2, 1))
    return np.matmul(GRAD_REF[None, :, :], inv_J)

def assemble_stiffness_matrix(areas, inv_J_T, nu, row_indices, col_indices, num_nodes):
    # The inverse transpose of the Jacobian J^-T transforms reference gradients to actual gradients.
    # B = J^-T ∇_{ξ,η} [N1, N2, N3] where N1, N2, N3 are the shape functions for the reference triangle.
    B = compute_physical_shape_gradients(inv_J_T)
    Ke_local = nu * np.einsum('eia,eja,e->eij', B, B, areas)

    data = Ke_local.reshape(-1)

    K = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    return K

def assemble_convection_matrix(U_k, V_k, areas, inv_J_T, elems, row_indices, col_indices, num_nodes):
    # Use 3-point triangle quadrature to avoid rank-deficient local convection rows.
    B = compute_physical_shape_gradients(inv_J_T)  # (num_elems, 3, 2)
    Ue = U_k[elems]  # (num_elems, 3)
    Ve = V_k[elems]  # (num_elems, 3)

    # Quadrature points in barycentric coordinates and weights on reference triangle.
    # Physical contribution uses area = |J| / 2, so weights sum to 1 over each element area.
    Nq = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    wq = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    Ce_local = np.zeros((elems.shape[0], 3, 3), dtype=np.float64)
    for q in range(3):
        Nvals = Nq[q]  # (3,)
        U_q = np.einsum('ei,i->e', Ue, Nvals)  # (num_elems,)
        V_q = np.einsum('ei,i->e', Ve, Nvals)  # (num_elems,)
        velocity_q = np.stack([U_q, V_q], axis=1)  # (num_elems, 2)
        adv_dot_grad = np.einsum('ea,eja->ej', velocity_q, B)  # (num_elems, 3)
        Ce_local += (areas * wq[q])[:, None, None] * np.einsum('i,ej->eij', Nvals, adv_dot_grad)
    
    data = Ce_local.reshape(-1)

    C = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    return C

def assemble_streamline_diffusion_matrix(U_k, V_k, areas, inv_J_T, elems, row_indices, col_indices, num_nodes, nu):
    B = compute_physical_shape_gradients(inv_J_T)
    U_centroid = U_k[elems].mean(axis=1)
    V_centroid = V_k[elems].mean(axis=1)
    velocity_centroid = np.stack([U_centroid, V_centroid], axis=1)
    speed = np.linalg.norm(velocity_centroid, axis=1)

    # Use element-wise length scale in SUPG parameter for unstructured meshes.
    h_e = np.sqrt(2.0 * areas)
    tau = ((2.0 * speed / h_e)**2 + (4.0 * nu / h_e**2)**2 + 1.0e-30)**(-0.5)
    streamline_derivative = np.einsum('ea,eia->ei', velocity_centroid, B)
    Se_local = tau[:, None, None] * areas[:, None, None] * np.einsum(
        'ei,ej->eij', streamline_derivative, streamline_derivative
    )

    data = Se_local.reshape(-1)

    S = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    return S

def apply_dirichlet(A, bc_indices):
    A_bc = A.tolil(copy=True)

    for idx in bc_indices:
        A_bc[:, idx] = 0.0
        A_bc[idx, :] = 0.0
        A_bc[idx, idx] = 1.0

    return A_bc.tocsr()

def initial_condition(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    U0 = np.sin(np.pi * x) * np.cos(np.pi * y)
    V0 = np.cos(np.pi * x) * np.sin(np.pi * y)
    return U0, V0

def dump_solution(coords, U, V, output_file):
    with open(output_file, 'w') as f:
        for i in range(coords.shape[0]):
            f.write(f"{coords[i, 0]:.6f}, {coords[i, 1]:.6f}, {U[i]:.6f}, {V[i]:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2-D Lid-Driven Cavity Flow with Finite Element Method')
    parser.add_argument('-n', '--num_elems', type=int, default=10, help='Number of elems along one dimension (default: 10)')
    parser.add_argument('-t', '--time_steps', type=int, default=100, help='Number of time steps to simulate (default: 100)')
    parser.add_argument('-d', '--time_step_size', type=float, default=0.01, help='Time step size (default: 0.01)')
    parser.add_argument('-m', '--mesh_directory', type=str, default='mesh', help='Directory containing mesh files (default: mesh)')
    parser.add_argument('-r', '--reynolds_number', type=float, default=100.0, help='Reynolds number for the simulation (default: 100.0)')
    parser.add_argument('--plot_every', type=int, default=10, help='Save solution every N time steps (default: 10)')
    parser.add_argument('-o', '--output_directory', type=str, default='output', help='Directory to save output plots (default: output)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    N = args.num_elems
    Nt = args.time_steps
    dt = args.time_step_size
    mesh_directory = args.mesh_directory
    Re = args.reynolds_number
    plot_every = args.plot_every
    output_directory = args.output_directory
    verbose = args.verbose

    main()