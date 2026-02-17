import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def plot_csv():
    # Create and format the plot
    plt.figure(figsize=(8, 6))
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for file, slope_i in zip(file_path, power_law_slope):
        # Load the data
        df = pd.read_csv(file)
        
        # Identify columns
        if num_rows:
            x = df.iloc[:num_rows, x_col]  # Column specified by x_col
            y = df.iloc[:num_rows, y_col]  # Column specified by y_col
        else:
            x = df.iloc[:, x_col]  # Column specified by x_col
            y = df.iloc[:, y_col]  # Column specified by y_col

        label = file.split("/")[-1].replace(".csv", "").replace("_", " ").title()
        plt.plot(x, y, label=label)
        plt.xlabel(df.columns[x_col], fontsize=14)
        plt.ylabel(df.columns[y_col], fontsize=14)

        # Plot reference power law slope if provided
        if not(math.isnan(slope_i)):
            x_ref = np.array([x.min(), x.max()])
            y_ref = (x_ref / x_ref[0]) ** slope_i * y.iloc[0]  # Scale to match first point
            plt.plot(x_ref, y_ref, label=f'Reference slope: {slope_i}', linestyle='dashed')
        plt.legend(fontsize=12)

    # Save and show
    if output_filename is not None:
        plt.savefig(output_filename)
        print(f"Plot saved as {output_filename}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
    parser.add_argument('-i', '--file_path', nargs='+', type=str, help='Path to the CSV file to plot')
    parser.add_argument('-t', '--title', type=str, default='Plot from CSV', help='Title of the plot')
    parser.add_argument('-x', type=int, default=0, help='Column index for x-axis (default: 0)')
    parser.add_argument('-y', type=int, default=1, help='Column index for y-axis (default: 1)')
    parser.add_argument('-l', '--log_x', action='store_true', help='Use log scale for x-axis')
    parser.add_argument('-L', '--log_y', action='store_true', help='Use log scale for y-axis')
    parser.add_argument('-o', '--output', type=str, help='Output filename for the plot (optional)')
    parser.add_argument('-n', '--num_lines', type=int, default=None, help='Number of lines to plot (default: 5)')
    parser.add_argument('-p', '--power_law_slope', nargs='+', type=float, help='Reference slope to plot in log-log plot (optional)')
    args = parser.parse_args()

    x_col = args.x
    y_col = args.y
    num_rows = args.num_lines
    file_path = args.file_path
    title = args.title
    log_x = args.log_x
    log_y = args.log_y
    power_law_slope = args.power_law_slope
    output_filename = args.output

    plot_csv()