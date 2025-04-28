import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots

class BrownianParticleSimulator:
    def reset(self):
        """
        Initialize particle positions randomly within bounds
        """
        self.positions = np.random.rand(self.num_particles, 3)
        self.positions[:, 0] = self.positions[:, 0] * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.positions[:, 1] = self.positions[:, 1] * (self.bounds[3] - self.bounds[2]) + self.bounds[2]
        self.positions[:, 2] = self.positions[:, 2] * (self.bounds[5] - self.bounds[4]) + self.bounds[4]

        # Store full trajectories
        self.trajectories = np.zeros((self.num_particles, self.total_frames, 3))
        self.trajectories[:, 0, :] = self.positions.copy()
        self.current_frame = 1

    def __init__(
        self, num_particles=3, duration=100, fps=60, temperature=300, viscosity=0.001, particle_radius=5e-7,
        bounds=None, drift=None
    ):
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)

        # Simulation parameters
        self.num_particles = num_particles
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        self.dt = 1 / fps

        # Physical parameters
        self.T = temperature
        self.eta = viscosity
        self.radius = particle_radius

        # Calculate diffusion coefficient (D)
        self.D = self.k_B * self.T / (6 * np.pi * self.eta * self.radius)  # m^2/s
        self.D *= 1e12  # to um^2/s

        # Bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        self.bounds = bounds if bounds is not None else [0, 100, 0, 100, 0, 100]

        # Drift velocity
        self.drift = np.array(drift) * self.dt if drift is not None else np.zeros(3)

        # Initialize
        self.reset()

    def step(self):
        """
        Advance the simulation by one time step
        """
        displacements = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.num_particles, 3)
        displacements += self.drift

        new_positions = self.positions + displacements

        # Reflective boundaries
        for i in range(self.num_particles):
            for j in range(3):
                if new_positions[i, j] < self.bounds[2 * j]:
                    new_positions[i, j] = 2 * self.bounds[2 * j] - new_positions[i, j]
                elif new_positions[i, j] > self.bounds[2 * j + 1]:
                    new_positions[i, j] = 2 * self.bounds[2 * j + 1] - new_positions[i, j]

        self.positions = new_positions

        if self.current_frame < self.total_frames:
            self.trajectories[:, self.current_frame, :] = self.positions.copy()
            self.current_frame += 1

    def simulate(self):
        """
        Run the full simulation
        """
        for _ in range(1, self.total_frames):
            self.step()
        return self.trajectories

    def plot_trajectories(self):
        """Plot all trajectories in 3D"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.viridis(np.linspace(0, 1, self.num_particles))

        for i in range(self.num_particles):
            x = self.trajectories[i, :, 0]
            y = self.trajectories[i, :, 1]
            z = self.trajectories[i, :, 2]
            ax.plot(x, y, z, '-', color=colors[i], alpha=0.7, label=f'Particle {i + 1}')
            ax.scatter(x[0], y[0], z[0], color=colors[i], marker='o')
            ax.scatter(x[-1], y[-1], z[-1], color=colors[i], marker='x')

        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_zlim(self.bounds[4], self.bounds[5])
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title('3D Brownian Motion Trajectories')
        ax.legend()
        plt.show()


# ===== Example usage =====
if __name__ == "__main__":
    sim = BrownianParticleSimulator(
        num_particles=3,
        duration=100,
        fps=30,
        temperature=300,
        viscosity=0.001,
        particle_radius=2.5e-7,
        bounds=[0, 50, 0, 50, 0, 50],
        drift=[0.1, 0.05, 0.02]
    )

    trajectories = sim.simulate()
    sim.plot_trajectories()
