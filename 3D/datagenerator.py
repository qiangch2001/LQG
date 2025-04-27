import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BrownianParticleSimulator3D:
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
        self, num_particles=5, duration=10, fps=30, temperature=300, viscosity=0.001, particle_radius=0.1e-6,
        bounds=None, drift=None
    ):
        """
        Initialize the Brownian motion simulator in 3D

        Parameters:
        - num_particles: Number of particles to simulate
        - duration: Total simulation time in seconds
        - fps: Frames per second for animation
        - temperature: Temperature in Kelvin
        - viscosity: Fluid viscosity in Pa·s (water ≈ 0.001)
        - particle_radius: Particle radius in meters
        - bounds: [xmin, xmax, ymin, ymax, zmin, zmax] in micrometers
        - drift: Optional drift velocity [vx, vy, vz] in μm/s
        """
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

        # Calculate diffusion coefficient (D) from Stokes-Einstein equation
        self.D = self.k_B * self.T / (6 * np.pi * self.eta * self.radius)  # in m²/s
        self.D *= 1e12  # convert to μm²/s

        # Set bounds (default 100×100×100 μm field of view)
        self.bounds = bounds if bounds is not None else [0, 100, 0, 100, 0, 100]

        # Set drift (optional)
        self.drift = np.array(drift) * self.dt if drift is not None else np.zeros(3)

        # Initialize positions
        self.reset()

    def step(self):
        """
        Advance the simulation by one time step
        """
        # Random displacements (Brownian motion)
        displacements = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.num_particles, 3)

        # Add drift if specified
        displacements += self.drift

        # Update positions
        new_positions = self.positions + displacements

        # Apply boundary conditions (reflective)
        for i in range(self.num_particles):
            for j in range(3):
                if new_positions[i, j] < self.bounds[2 * j]:
                    new_positions[i, j] = 2 * self.bounds[2 * j] - new_positions[i, j]
                elif new_positions[i, j] > self.bounds[2 * j + 1]:
                    new_positions[i, j] = 2 * self.bounds[2 * j + 1] - new_positions[i, j]

        self.positions = new_positions

        # Store trajectory
        if self.current_frame < self.total_frames:
            self.trajectories[:, self.current_frame, :] = self.positions.copy()
            self.current_frame += 1

    def simulate(self):
        """
        Run complete simulation
        """
        for _ in range(1, self.total_frames):
            self.step()
        return self.trajectories

    def plot_trajectories(self):
        """Plot all trajectories on a 3D figure"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.viridis(np.linspace(0, 1, self.num_particles))

        for i in range(self.num_particles):
            x = self.trajectories[i, :, 0]
            y = self.trajectories[i, :, 1]
            z = self.trajectories[i, :, 2]
            ax.plot(x, y, z, '-', color=colors[i], alpha=0.7, label=f'Particle {i + 1}')
            ax.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker='o')
            ax.scatter(x[-1], y[-1], z[-1], color=colors[i], s=100, marker='x')

        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_zlim(self.bounds[4], self.bounds[5])
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_zlabel('Z position (μm)')
        ax.set_title('3D Brownian Motion Trajectories')
        ax.legend()
        plt.show()

    def animate(self, trajectories=None):
        """Create an animation of the particle motion in 3D"""
        if trajectories is None:
            trajectories = self.trajectories

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_zlim(self.bounds[4], self.bounds[5])
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_zlabel('Z position (μm)')
        ax.set_title('3D Brownian Motion Simulation')

        colors = plt.cm.viridis(np.linspace(0, 1, self.num_particles))
        particles = [ax.plot([], [], [], 'o', color=colors[i], markersize=8,
                             label=f'Particle {i + 1}')[0] for i in range(self.num_particles)]
        paths = [ax.plot([], [], [], '-', color=colors[i], alpha=0.5, linewidth=1)[0]
                 for i in range(self.num_particles)]

        def init():
            for particle, path in zip(particles, paths):
                particle.set_data([], [])
                particle.set_3d_properties([])
                path.set_data([], [])
                path.set_3d_properties([])
            return particles + paths

        def update(frame):
            for i, (particle, path) in enumerate(zip(particles, paths)):
                x = trajectories[i, :frame, 0]
                y = trajectories[i, :frame, 1]
                z = trajectories[i, :frame, 2]
                particle.set_data(x[-1:], y[-1:])
                particle.set_3d_properties(z[-1:])
                path.set_data(x, y)
                path.set_3d_properties(z)
            return particles + paths

        ani = FuncAnimation(fig, update, frames=self.total_frames, init_func=init, blit=True, interval=50)
        plt.legend()
        return ani

if __name__ == "__main__":
    # Example usage
    sim = BrownianParticleSimulator3D(
        num_particles=1,
        duration=5,  # 5 seconds simulation
        fps=20,  # 20 frames per second
        temperature=300,  # Room temperature (300K)
        viscosity=0.001,  # Water viscosity (0.001 Pa·s)
        particle_radius=0.1e-6,  # 100 nm particles
        bounds=[0, 20, 0, 20, 0, 20],  # 50×50×50 μm field of view
        drift=[0.1, 0.05, 0.02]  # Drift in x, y, z (μm/s)
    )

    # Run simulation
    trajectories = sim.simulate()

    # Display animation
    sim.plot_trajectories()
