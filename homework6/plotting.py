import matplotlib.pyplot as plt
import numpy as np

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
    # Slice the interior (ignore ghost cells)
    u_int = u[1:-1, 1:-1]
    v_int = v[1:-1, 1:-1]
    x_int = x[1:-1, 1:-1]
    y_int = y[1:-1, 1:-1]

    # Create the plot using figure and axes objects
    fig, ax = plt.subplots(figsize=(6, 5))

    # Calculate velocity magnitude for line coloring
    speed = np.sqrt(u_int**2 + v_int**2)

    # Plot streamlines on the axes
    strm = ax.streamplot(x_int, y_int, u_int, v_int, color=speed, cmap='viridis', 
                         linewidth=1.5, density=1.2)

    fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.tight_layout()
    fig.savefig(title)
    plt.close(fig)