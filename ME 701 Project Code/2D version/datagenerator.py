import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BrownianParticleSimulator:
    def reset(self):
        """
        Initialize particle positions randomly within bounds
        """
        self.positions = np.random.rand(self.num_particles, 2)
        self.positions[:, 0] = self.positions[:, 0] * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.positions[:, 1] = self.positions[:, 1] * (self.bounds[3] - self.bounds[2]) + self.bounds[2]

        # Store full trajectories
        self.trajectories = np.zeros((self.num_particles, self.total_frames, 2))
        self.trajectories[:, 0, :] = self.positions.copy()
        self.current_frame = 1

    def __init__(
        self, num_particles=3, duration=10, fps=30, temperature=300, viscosity=0.001, particle_radius=0.1e-6,
        bounds=None, drift=None
    ):
        """
        Initialize the Brownian motion simulator

        Parameters:
        - num_particles: Number of particles to simulate
        - duration: Total simulation time in seconds
        - fps: Frames per second for animation
        - temperature: Temperature in Kelvin
        - viscosity: Fluid viscosity in Pa·s (water ≈ 0.001)
        - particle_radius: Particle radius in meters
        - bounds: [xmin, xmax, ymin, ymax] in micrometers
        - drift: Optional drift velocity [vx, vy] in μm/s
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

        print("D - Coeff: ", self.D)

        # Set bounds (default 100×100 μm field of view)
        self.bounds = bounds if bounds is not None else [0, 100, 0, 100]

        # Set drift (optional)
        self.drift = np.array(drift) * self.dt if drift is not None else np.zeros(2)

        # Initialize positions
        self.reset()

    def step(self):
        """
        Advance the simulation by one time step
        """
        # Random displacements (Brownian motion)
        displacements = np.sqrt(2 * self.D * self.dt) * np.random.randn(self.num_particles, 2)

        # Add drift if specified
        displacements += self.drift

        # Update positions
        new_positions = self.positions + displacements

        # Apply boundary conditions (reflective)
        for i in range(self.num_particles):
            for j in range(2):
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
        """Plot all trajectories on single figure"""
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_particles))

        for i in range(self.num_particles):
            x = self.trajectories[i, :, 0]
            y = self.trajectories[i, :, 1]
            plt.plot(x, y, '-', color=colors[i], alpha=0.7, label=f'Particle {i + 1}')
            plt.plot(x[0], y[0], 'o', color=colors[i], markersize=8)
            plt.plot(x[-1], y[-1], 'x', color=colors[i], markersize=8)

        plt.xlim(self.bounds[0], self.bounds[1])
        plt.ylim(self.bounds[2], self.bounds[3])
        plt.xlabel('X position (μm)')
        plt.ylabel('Y position (μm)')
        plt.title('Brownian Motion Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    def animate(self, trajectories=None):
        """Create an animation of the particle motion"""
        if trajectories is None:
            trajectories = self.trajectories

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_title('2D Brownian Motion Simulation')

        colors = plt.cm.viridis(np.linspace(0, 1, self.num_particles))
        particles = [ax.plot([], [], 'o', color=colors[i], markersize=8,
                             label=f'Particle {i + 1}')[0] for i in range(self.num_particles)]
        paths = [ax.plot([], [], '-', color=colors[i], alpha=0.5, linewidth=1)[0]
                 for i in range(self.num_particles)]

        def init():
            for particle, path in zip(particles, paths):
                particle.set_data([], [])
                path.set_data([], [])
            return particles + paths

        def update(frame):
            for i, (particle, path) in enumerate(zip(particles, paths)):
                x = trajectories[i, :frame, 0]
                y = trajectories[i, :frame, 1]
                particle.set_data(trajectories[i, frame, 0], trajectories[i, frame, 1])
                path.set_data(x, y)
            return particles + paths

        ani = FuncAnimation(fig, update, frames=self.total_frames, init_func=init, blit=True, interval=50)
        plt.legend()
        return ani


# Example usage
if __name__ == "__main__":
    sim = BrownianParticleSimulator(
        num_particles=5,
        duration=5,  # 30 second simulation
        fps=20,  # 20 frames per second
        temperature=300,  # Room temperature (300K)
        viscosity=0.001,  # Water viscosity (0.001 Pa·s)
        particle_radius=0.1e-6,  # 100 nm particles
        bounds=[0, 50, 0, 50],  # 50×50 μm field of view
        drift=[0.1, 0.05]  # Slight drift in x and y (μm/s)
    )

    # Run simulation
    trajectories = sim.simulate()

    # Display animation
    sim.plot_trajectories()