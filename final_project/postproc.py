import numpy as np
import argparse

def main():
    nu = 1.0 / Re  # Kinematic viscosity

    # Load all solution_*.txt files onto U and V arrays
    file = f"{input_dir}/solution_{t:.4f}.txt"
    x, y, U, V = np.loadtxt(file, delimiter=",", unpack=True)

    # Get x and y indices for the 9 points of interest
    interest_pts = [(0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)]
    indices = []
    for pt in interest_pts:
        idx = np.argmin(np.sqrt((x - pt[0])**2 + (y - pt[1])**2))
        indices.append(idx)

    # Get literature solution from https://doi.org/10.1016/j.camwa.2019.08.036
    if t == 0.5 and Re == 1000.0:
        analytical_U = [0.29673, 0.57601, 0.29673, 0.27288, 0.00000, -0.27288, -0.29673, -0.57601, -0.29673]
    elif t == 1.0 and Re == 1000.0:
        analytical_U = [0.18824, 0.37442, 0.18824, 0.18529, 0.00000, -0.18529, -0.18824, -0.37442, -0.18824]
    elif t == 0.5 and Re == 10000.0:
        analytical_U = [0.29725, 0.57738, 0.29725, 0.27382, 0.00000, -0.27382, -0.29725, -0.57738, -0.29725]
    elif t == 1.0 and Re == 10000.0:
        analytical_U = [0.18846, 0.37491, 0.18846, 0.18558, 0.00000, -0.18558, -0.18846, -0.37491, -0.18846]
    else:
        print("No analytical solution available for this time point.")
        return
    
    # Compare with literature solution
    RMSE = 0.0
    for i, idx in enumerate(indices):
        RMSE += (U[idx] - analytical_U[i])**2
        print(f"Point {interest_pts[i]}: U={U[idx]:.4f}, analytical_U={analytical_U[i]:.4f}")
    RMSE = np.sqrt(RMSE / len(indices))
    print(f"Root Mean Square Error compared to literature solution: {RMSE:.4e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process simulation data.')
    parser.add_argument('-n', '--num_elems', type=int, default=10, help='Number of elems along one dimension (default: 10)')
    parser.add_argument('-i', '--input_directory', type=str, default='output', help='Directory containing the simulation data files.')
    parser.add_argument('-o', '--output_directory', type=str, default='output', help='Directory to save the processed results.')
    parser.add_argument('-r', '--reynolds_number', type=float, default=100.0, help='Reynolds number for the simulation.')
    parser.add_argument('-t', '--time_point', type=float, default=1.0, help='Time point to visualize (e.g., 0.1 for t=0.1s).')
    args = parser.parse_args()

    N = args.num_elems
    input_dir = args.input_directory
    output_dir = args.output_directory
    Re = args.reynolds_number
    t = args.time_point

    main()