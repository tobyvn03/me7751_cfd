import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 128
Re = 400.00
u_lid = 0.1
max_iter = 200000
tol = 1e-6

# D2Q9 lattice velocities
cx = np.array([0,  1,  1, 0, -1, -1, -1,  0,  1], dtype=float)
cy = np.array([0,  0,  1, 1,  1,  0, -1, -1, -1], dtype=float)

# Corresponding weights
W  = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# Lattice speed of sound squared
CS2 = 1.0 / 3.0

OPPOSITE = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])

N_DIRS = 9

def compute_tau(Re, N, u_lid):
    nu_lattice = u_lid * N / Re          # lattice kinematic viscosity
    tau = nu_lattice / CS2 + 0.5
    return tau


def equilibrium(rho, ux, uy):
    feq = np.zeros((N_DIRS, *rho.shape))
    u2  = ux**2 + uy**2

    for i in range(N_DIRS):
        cu  = cx[i] * ux + cy[i] * uy
        feq[i] = W[i] * rho * (1.0 + cu/CS2 + cu**2/(2.0*CS2**2) - u2/(2.0 * CS2))
    return feq


def flow_vars(f):
    rho = np.sum(f, axis=0)
    ux  = np.sum(f * cx[:, None, None], axis=0)/rho
    uy  = np.sum(f * cy[:, None, None], axis=0)/rho

    ux[:,  0] = 0.0;    uy[:,  0] = 0.0   # bottom wall (stationary)
    ux[:, -1] = u_lid;  uy[:, -1] = 0.0   # top wall (moving lid)
    ux[0,  :] = 0.0;    uy[0,  :] = 0.0   # left wall (stationary)
    ux[-1, :] = 0.0;    uy[-1, :] = 0.0   # right wall (stationary)
    return rho, ux, uy

def stream(f_post):
    Nx, Ny = f_post.shape[1], f_post.shape[2]
    f_new  = np.zeros_like(f_post)

    for i in range(N_DIRS):
        ex = int(cx[i])
        ey = int(cy[i])

        # for x in range(Nx):
        #     for y in range(Ny):
        #         # destination node
        #         xd = x + ex
        #         yd = y + ey
        #         if 0 <= xd < Nx and 0 <= yd < Ny:
        #             f_new[i, xd, yd] = f_post[i, x, y]

        # Using slicing for faster streaming

        # x-axis
        if ex == 1:           # streams rightward
            src_x  = slice(0,  Nx - 1)
            dst_x  = slice(1,  Nx)
        elif ex == -1:        # streams leftward
            src_x  = slice(1,  Nx)
            dst_x  = slice(0,  Nx - 1)
        else:                 # no x movement
            src_x  = slice(0,  Nx)
            dst_x  = slice(0,  Nx)

        # y-axis
        if ey == 1:           # streams upward
            src_y  = slice(0,  Ny - 1)
            dst_y  = slice(1,  Ny)
        elif ey == -1:        # streams downward
            src_y  = slice(1,  Ny)
            dst_y  = slice(0,  Ny - 1)
        else:                 # no y movement
            src_y  = slice(0,  Ny)
            dst_y  = slice(0,  Ny)

        f_new[i, dst_x, dst_y] = f_post[i, src_x, src_y]

    return f_new

def convergence_check(ux_new, uy_new, ux_old, uy_old):
    du2   = (ux_new - ux_old)**2 + (uy_new - uy_old)**2
    u2    = ux_new**2 + uy_new**2
    denom = np.sqrt(np.sum(u2))
    if denom < 1e-30:
        return 1.0  # avoid division by zero at startup
    return np.sqrt(np.sum(du2)) / denom

def apply_boundary_conditions(f, f_post, u_lid, rho):

    Nx, Ny = f_post.shape[1], f_post.shape[2]

    # Bottom wall: y = 0
    for i in [6, 7, 8]:
        f[OPPOSITE[i], :, 0] = f_post[i, :, 0]

    # Top wall: y = Ny-1
    for i in [2, 3, 4]:
        momentum_correction = (2.0 * W[i] * rho[:, Ny-1]* (cx[i] * u_lid) / CS2)
        f[OPPOSITE[i], :, Ny-1] = (f_post[i, :, Ny-1] - momentum_correction)
        
    # Left wall: x = 0
    for i in [4, 5, 6]:
        f[OPPOSITE[i], 0, :] = f_post[i, 0, :]

    # Right wall: x = Nx-1
    for i in [1, 2, 8]:
        f[OPPOSITE[i], Nx-1, :] = f_post[i, Nx-1, :]

def check_divergence(u, v, dx, dy):
    
    div = np.zeros((N, N))
    for j in range(1,N):
        for i in range(1,N):
            div[i-1, j-1] = (u[i, j] - u[i-1, j]) / dx + (v[i, j] - v[i, j-1]) / dy
    return div

# MAIN SOLVER

# Parameters
Nx, Ny = N, N
tau    = compute_tau(Re, N, u_lid)
print(f"Re={Re}, N={N}, u_lid={u_lid:.3f}, tau={tau:.4f}")
print(f"  Ma = {u_lid / np.sqrt(CS2):.4f}")

# Initialise: uniform density, zero velocity
rho = np.ones((Nx, Ny))
ux  = np.zeros((Nx, Ny))
uy  = np.zeros((Nx, Ny))
f   = equilibrium(rho, ux, uy)

# --- Time-stepping loop ---
for step in range(max_iter):

    ux_old, uy_old = ux.copy(), uy.copy()

    # 1) Flow variables
    rho, ux, uy = flow_vars(f)

    # 2) equilibrium distribution
    feq = equilibrium(rho, ux, uy)

    # 3) collision
    f_post = f - (1.0 / tau) * (f - feq)

    # 4) streaming
    f = stream(f_post)

    # 5) boundary conditions
    apply_boundary_conditions(f, f_post, u_lid, rho)

    # Convergence check
    err = convergence_check(ux, uy, ux_old, uy_old)
    print(f"  step {step:6d} | residual = {err:.3e}")
    if err < tol:
        Ma = u_lid / np.sqrt(CS2)
        print(f"  Ma = {Ma:.4f}, tau = {tau:.4f}")
        print(f"  Converged at step {step}.")
        break

# --- Post-processing ---

xc = np.linspace(0, 1, N)
yc = np.linspace(0, 1, N)
ux = ux/u_lid
uy = uy/u_lid

# steamline plot
plt.streamplot(xc, yc, ux.T, uy.T, density=2, linewidth=0.8)
plt.title(f"Streamlines Re={Re}")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("streamlines.png")
plt.show()

#divergence check
div = check_divergence(ux, uy, 1.0, 1.0)

fig, ax = plt.subplots(figsize=(6,6))
cf = ax.contourf(xc, yc, div.T, levels=50, cmap='RdBu_r')
plt.colorbar(cf, ax=ax, label='divergence')
ax.set_title(f'Divergence field Re={Re}, N={N}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.savefig('divergence.png')
plt.show()

print(f"\nMax divergence: {np.max(np.abs(div)):.2e}")

#centreline velocities
u_centre = ux[N//2, 0:N]
v_centre = uy[0:N, N//2]

np.savetxt(f"Ucentre_{N}.csv", u_centre, delimiter=",")
np.savetxt(f"Vcentre_{N}.csv", v_centre, delimiter=",")

# u along vertical centerline
plt.figure()
plt.plot(u_centre, yc)
plt.xlabel('u')
plt.ylabel('y')
plt.title(f'u at x=0.5, Re={Re}')
plt.savefig('u_centre.png')
plt.show()

# v along horizontal centerline
plt.figure()
plt.plot(xc, v_centre)
plt.xlabel('x')
plt.ylabel('v')
plt.title(f'v at y=0.5, Re={Re}')
plt.savefig('v_centre.png')
plt.show()