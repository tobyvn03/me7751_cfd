import numpy as np

def d_dx_backward(f, dx):
    df_dx = np.zeros_like(f)
    df_dx[:, 1:-1] = (f[:, 1:-1] - f[:, :-2]) / dx
    return df_dx

def d_dx_forward(f, dx):
    df_dx = np.zeros_like(f)
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, 1:-1]) / dx
    return df_dx

def d_dx_central(f, dx):
    df_dx = np.zeros_like(f)
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    return df_dx

def avg_x_backward(f):
    avg_f_x = np.zeros_like(f)
    avg_f_x[:, 1:-1] = (f[:, 1:-1] + f[:, :-2]) / 2
    return avg_f_x

def avg_x_forward(f):
    avg_f_x = np.zeros_like(f)
    avg_f_x[:, 1:-1] = (f[:, 2:] + f[:, 1:-1]) / 2
    return avg_f_x

def d_dy_backward(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[1:-1, :] = (f[1:-1, :] - f[:-2, :]) / dy
    return df_dy

def d_dy_forward(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[1:-1, :] = (f[2:, :] - f[1:-1, :]) / dy
    return df_dy

def d_dy_central(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
    return df_dy

def avg_y_backward(f):
    avg_f_y = np.zeros_like(f)
    avg_f_y[1:-1, :] = (f[1:-1, :] + f[:-2, :]) / 2
    return avg_f_y

def avg_y_forward(f):
    avg_f_y = np.zeros_like(f)
    avg_f_y[1:-1, :] = (f[2:, :] + f[1:-1, :]) / 2
    return avg_f_y

def d2_dx2(f, dx):
    d2f_dx2 = np.zeros_like(f)
    d2f_dx2[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (dx * dx)
    return d2f_dx2

def d2_dy2(f, dy):
    d2f_dy2 = np.zeros_like(f)
    d2f_dy2[1:-1, :] = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / (dy * dy)
    return d2f_dy2