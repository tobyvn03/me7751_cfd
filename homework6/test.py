from solver import *
from plotting import *
from finite_differences import *

def main():
    N = 10
    rho0 = 100
    indexes = range(9)
    Cx = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    Cy = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    tau = 0.6

    # Define ghost cell mask for applying boundary conditions
    ghost_mask = np.full((N+2, N+2), False)
    ghost_mask[0, :] = True
    ghost_mask[-1, :] = True
    ghost_mask[:, 0] = True
    ghost_mask[:, -1] = True

    F = np.ones((N+2, N+2, 9))
    np.random.seed(42)
    F += 0.01 * np.random.randn(N+2, N+2, 9)
    
    # rho = np.sum(F, axis=2)
    # for i in indexes:
    #     F[:, :, i] *= rho0 / rho
    F[ghost_mask, :] = 0
    
    # Free-streaming propagation step
    for t in range(1000):
        # Free-streaming propagation step
        for i in indexes:
            F[:, :, i] = np.roll(F[:, :, i], Cx[i], axis=1)
            F[:, :, i] = np.roll(F[:, :, i], Cy[i], axis=0)

        # Compute macroscopic variables
        rho = np.sum(F, axis=2)
        u = np.sum(F * Cx, axis=2) / rho
        v = np.sum(F * Cy, axis=2) / rho       

        # Collision step with BGK operator
        F_eq = np.zeros(F.shape)
        for i in indexes:
            F_eq[:, :, i] = weights[i] * rho * (1 + 3*(Cx[i]*u + Cy[i]*v) + 9/2 * (Cx[i]*u + Cy[i]*v)**2 - 3/2 * (u**2 + v**2))
        F -= (F - F_eq) / tau

        if t % 100 == 0:
            rho_new = np.sum(F, axis=2)
            u_new = np.sum(F * Cx, axis=2) / rho_new
            v_new = np.sum(F * Cy, axis=2) / rho_new

            divergence = d_dx_central(u_new, 1)[1:-1, 1:-1] + d_dy_central(v_new, 1)[1:-1, 1:-1]
            plot_heatmap(divergence, title=f'divergence_test')
            plot_heatmap(rho_new[1:-1, 1:-1] / 3, title=f'pressure_test')
            plot_heatmap(u_new[1:-1, 1:-1], title=f'xvelocity_test')
            plot_heatmap(v_new[1:-1, 1:-1], title=f'yvelocity_test')

if __name__ == "__main__":
    main()