import seaborn as sns
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np

from main import track_particles_lqg
from datagenerator import BrownianParticleSimulator

# Define the parameter sets
lambda_x_values = [0.5, 2, 8, 16]
lambda_u_values = [0.2, 0.5, 2, 8]

# Result matrix to store the mean RMSEs
mean_rmse_matrix = np.zeros((len(lambda_x_values), len(lambda_u_values)))


# Define a function to run a single simulation
def run_single_simulation(lambda_x, lambda_u):
    sim = BrownianParticleSimulator(
        num_particles=1,
        duration=5,
        fps=20,
        temperature=300,
        viscosity=0.001,
        particle_radius=0.1e-6,
        bounds=[0, 20, 0, 20],
        drift=[0.1, 0.05]
    )

    results = track_particles_lqg(
        sim,
        lambda_x=lambda_x,
        lambda_u=lambda_u,
        process_noise=0.1,
        measurement_noise=0.5
    )

    # Compute RMSE for the simulation
    errors = results['true_particles'] - results['true_stage']
    rmse = np.sqrt(np.mean(np.sum(errors ** 2, axis=2), axis=1))
    return np.mean(rmse)  # Single particle, so directly take the mean


# Start running simulations
for i, lambda_x in enumerate(lambda_x_values):
    for j, lambda_u in enumerate(lambda_u_values):
        rmses = []

        # Run 10 simulations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_single_simulation, lambda_x, lambda_u) for _ in range(100)]
            for future in concurrent.futures.as_completed(futures):
                rmses.append(future.result())

        mean_rmse = np.mean(rmses)
        mean_rmse_matrix[i, j] = mean_rmse
        print(f"lambda_x={lambda_x}, lambda_u={lambda_u}, Mean RMSE={mean_rmse:.3f} μm")

# Plot the results as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(mean_rmse_matrix, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=lambda_u_values, yticklabels=lambda_x_values)
plt.xlabel('lambda_u')
plt.ylabel('lambda_x')
plt.title('Mean RMSE for Different (lambda_x, lambda_u)')
plt.show()
