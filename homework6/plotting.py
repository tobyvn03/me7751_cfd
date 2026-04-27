import matplotlib.pyplot as plt
import numpy as np
from solver import create_grid

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

def plot_heatmap(field, title="Divergence"):
    """
    Plots a single heatmap for the provided scalar `field` (e.g. divergence).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(field, origin='lower', extent=[0, 1, 0, 1],
                   cmap='RdBu_r', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(title)
    plt.close(fig)

def plot_fluid_streamlines(u, v, x, y, title="Velocity Streamlines"):
    """
    Plots velocity streamlines for the interior fluid cells.
    """
    # Create the plot using figure and axes objects
    fig, ax = plt.subplots(figsize=(6, 5))

    # Calculate velocity magnitude for line coloring
    speed = np.sqrt(u**2 + v**2)

    # Plot streamlines on the axes
    strm = ax.streamplot(x, y, u, v, color=speed, cmap='viridis', 
                         linewidth=1.5, density=1.2)

    fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.tight_layout()
    fig.savefig(title)
    plt.close(fig)

def plot_analytical(Re, trial, axes_u, axes_v):
    (x, y) = create_grid(129)
    x = x.astype(float)
    y = y.astype(float)
    x /= 128.0
    y /= 128.0
    x_indices = np.array([1, 9, 10, 11, 13, 21, 30, 31, 65, 104, 111, 117, 122, 123, 124, 125, 129]) - 1
    y_indices = np.array([1, 8, 9, 10, 14, 23, 37, 59, 65, 80, 95, 110, 123, 124, 125, 126, 129]) - 1
    if Re == 100:
        v_analytical = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])
        u_analytical = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
    elif Re == 400:
        v_analytical = np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000])
        u_analytical = np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000])
    else:
        raise ValueError(f"Analytical solution not available for Re={Re}")
    axes_v.plot(x[0, x_indices], v_analytical, 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")
    axes_u.plot(u_analytical, y[y_indices, 0], 'x', color=f'C{trial-1}', label=f"Analytical: Trial {trial}")