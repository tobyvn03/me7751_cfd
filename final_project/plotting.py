import matplotlib.pyplot as plt

def plot_solution(coords, U, V, output_file):
    plt.figure(figsize=(8, 8))
    plt.quiver(coords[:, 0], coords[:, 1], U, V)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Field')
    plt.axis('equal')
    plt.savefig(output_file)
    plt.close()

def plot_field(coords, field, output_file, title='Field'):
    plt.figure(figsize=(8, 8))
    plt.tricontourf(coords[:, 0], coords[:, 1], field, levels=50, cmap='viridis')
    plt.colorbar(label=title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.axis('equal')
    plt.savefig(output_file)
    plt.close()

