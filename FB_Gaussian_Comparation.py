import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Set fixed random seed for repeatability
np.random.seed(42)

# Simulation setup
true_position = (10e-9, 15e-9)
num_points = 5
circle_radius = 100e-9
snr_levels = np.linspace(3.28, 18.5, 100)
num_trials = 100
sigma_x = sigma_y = 203e-9

def generate_circle_points(center, radius, num_points): 

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_positions = center[0] + radius * np.cos(angles)
    y_positions = center[1] + radius * np.sin(angles)

    return x_positions, y_positions

x_positions, y_positions = generate_circle_points(true_position, circle_radius, num_points)

# FluoroBancroft implementation
def fluoro_bancroft(x, y, intensity, sigma_x, sigma_y, background_intensity):

    valid_intensity = np.maximum(intensity - background_intensity, 1e-6)
    P2 = 2 * sigma_x ** 2 * np.log(valid_intensity)
    alpha = 0.5 * (x ** 2 + y ** 2 + P2)
    B = np.column_stack((x, y, np.ones_like(x)))
    B_pseudo_inv = np.linalg.pinv(B)
    pos = B_pseudo_inv @ alpha
    Q = np.array([[1, 0], [0, sigma_x / sigma_y]])
    estimated_position = Q @ pos[:2]

    return estimated_position[0], estimated_position[1]

# Gaussian fitting with curve_fit
def gaussian_2d_model(xy, x0, y0, A):

    x, y = xy
    sigma = 203e-9

    return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def refined_gaussian_fitting(x, y, intensity):

    total_intensity = np.sum(intensity)

    if total_intensity <= 0:

        return np.mean(x), np.mean(y)
    
    x0_initial = np.sum(x * intensity) / total_intensity
    y0_initial = np.sum(y * intensity) / total_intensity
    A_initial = np.max(intensity)

    try:
        popt, _ = curve_fit(
            gaussian_2d_model,
            (x, y),
            intensity,
            p0=(x0_initial, y0_initial, A_initial),
            bounds=([x.min(), y.min(), 0], [x.max(), y.max(), np.inf])
        )
        return popt[0], popt[1]
    
    except:
        return x0_initial, y0_initial

# Containers for results
fb_x_std, fb_y_std = [], []
gf_x_std, gf_y_std = [], []

# Simulation loop
for snr in snr_levels:

    fb_xs, fb_ys = [], []
    gf_xs, gf_ys = [], []

    for _ in range(num_trials):

        dx = x_positions - true_position[0]
        dy = y_positions - true_position[1]
        spatial_factors = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_x ** 2))

        time_per_point = 5e-3 / num_points
        avg_spatial_factor = np.mean(spatial_factors)
        desired_signal_photons = snr ** 2
        photon_rate = desired_signal_photons / (time_per_point * avg_spatial_factor)
        signal_mean = photon_rate * time_per_point * spatial_factors

        background_rate = 200
        background_mean = background_rate * time_per_point
        signal_photons = np.random.poisson(signal_mean)
        background_photons = np.random.poisson(background_mean, size=num_points)
        intensity_values = signal_photons + background_photons

        try:
            x_fb, y_fb = fluoro_bancroft(x_positions, y_positions, intensity_values, sigma_x, sigma_y, background_mean)
            fb_xs.append((x_fb - true_position[0]) * 1e9)
            fb_ys.append((y_fb - true_position[1]) * 1e9)

        except:
            fb_xs.append(np.nan)
            fb_ys.append(np.nan)

        try:
            x_gf, y_gf = refined_gaussian_fitting(x_positions, y_positions, intensity_values)
            gf_xs.append((x_gf - true_position[0]) * 1e9)
            gf_ys.append((y_gf - true_position[1]) * 1e9)
            
        except:
            gf_xs.append(np.nan)
            gf_ys.append(np.nan)

    fb_x_std.append(np.nanstd(fb_xs))
    fb_y_std.append(np.nanstd(fb_ys))
    gf_x_std.append(np.nanstd(gf_xs))
    gf_y_std.append(np.nanstd(gf_ys))

# Plotting results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(snr_levels, gf_x_std, 'r--', label="Gaussian Fit (curve_fit)")
plt.plot(snr_levels, fb_x_std, 'b-', label="FluoroBancroft")
plt.title("Std. Dev. of X Position")
plt.xlabel("SNR")
plt.ylabel("Std. Dev. (nm)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(snr_levels, gf_y_std, 'r--', label="Gaussian Fit (curve_fit)")
plt.plot(snr_levels, fb_y_std, 'b-', label="FluoroBancroft")
plt.title("Std. Dev. of Y Position")
plt.xlabel("SNR")
plt.ylabel("Std. Dev. (nm)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
