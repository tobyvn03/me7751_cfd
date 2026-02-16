import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_csv(x_col, y_col, num_rows, file_path, title='Plot from CSV', semilog=False, output_filename=None):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Identify columns
    x = df.iloc[:num_rows, x_col]  # Column specified by x_col
    y = df.iloc[:num_rows, y_col]  # Column specified by y_col
        
    # Create the plot
    plt.plot(x, y)
    if semilog:
        plt.yscale('log')
        plt.xscale('log')
    
    # Formatting
    plt.title(title)
    plt.xlabel(df.columns[x_col])
    plt.ylabel(df.columns[y_col])
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save and show
    if output_filename is not None:
        plt.savefig(output_filename)
        plt.savefig(output_filename)
        print(f"Plot saved as {output_filename}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
    parser.add_argument('-f', '--file_path', type=str, help='Path to the CSV file to plot')
    parser.add_argument('-t', '--title', type=str, default='Plot from CSV', help='Title of the plot')
    parser.add_argument('-x', type=int, default=0, help='Column index for x-axis (default: 0)')
    parser.add_argument('-y', type=int, default=1, help='Column index for y-axis (default: 1)')
    parser.add_argument('-l', '--semilog', action='store_true', help='Use semilog scale for y-axis')
    parser.add_argument('-o', '--output', type=str, help='Output filename for the plot (optional)')
    parser.add_argument('-n', '--num_lines', type=int, default=5, help='Number of lines to plot (default: 5)')
    args = parser.parse_args()
    plot_csv(args.x, args.y, args.num_lines, args.file_path, args.title, args.semilog, args.output)