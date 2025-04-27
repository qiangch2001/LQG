import numpy as np
from datagenerator import BrownianParticleSimulator3D

class LQGController:
    """
    Implements LQG: Kalman Filter state estimator combined with LQR control.
    """
    def __init__(self, A, B, H, Q_kf, R_kf, lambda_x=1.0, lambda_u=0.1):
        # System matrices
        self.A = A                # State transition (Ao)
        self.B = B                # Control input (Bo)
        self.H = H                # Measurement matrix
        # Kalman filter covariances
        self.Q_kf = Q_kf          # Process noise
        self.R_kf = R_kf          # Measurement noise
        # LQR weights
        self.lambda_x = lambda_x  # state error weight
        self.lambda_u = lambda_u  # control effort weight

        # Dimensions
        self.n = A.shape[0]
        self.m = B.shape[1]

        # Precompute LQR gain
        self.G = self._compute_lqr_gain()

        # Initialize KF state and covariance
        self.x_hat = np.zeros((self.n,))
        self.P = np.eye(self.n)

    def _compute_lqr_gain(self, max_iters=1000, tol=1e-6):
        # Build Q_lqr and R_lqr for full state
        Hx = self.H
        Q_lqr = self.lambda_x * (Hx.T @ Hx)
        R_lqr = np.eye(self.m) * self.lambda_u

        P = np.eye(self.n)
        for _ in range(max_iters):
            BT_P = self.B.T @ P
            inv_term = np.linalg.inv(R_lqr + BT_P @ self.B)
            P_next = Q_lqr + self.A.T @ P @ self.A - self.A.T @ P @ self.B @ inv_term @ BT_P @ self.A
            if np.max(np.abs(P_next - P)) < tol:
                P = P_next
                break
            P = P_next
        return inv_term @ BT_P @ self.A

    def predict(self, u=None):
        if u is None:
            u = np.zeros(self.m)
        self.x_hat = self.A @ self.x_hat + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q_kf

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R_kf
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x_hat
        self.x_hat = self.x_hat + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

    def control(self):
        return -self.G @ self.x_hat


def track_particles_lqg(simulator, lambda_x=1.0, lambda_u=0.1,
                         process_noise=0.1, measurement_noise=0.5):
    true_traj = simulator.simulate()
    N, T, _ = true_traj.shape

    # Storage
    results = {key: np.zeros_like(true_traj) for key in [
        'true_particles', 'true_stage',
        'measured', 'est_particles', 'est_stage'
    ]}
    results['control_inputs'] = np.zeros((N, T, 3))  # 3D control inputs

    # Initialize stage positions to match the initial particle positions
    stage_pos = true_traj[:, 0, :3]  # Set the initial stage position to the initial particle position
    dt = simulator.dt

    # System matrices for 3D
    Ao = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, dt, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, dt, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    Bo = np.array([[dt, 0, 0], [0, dt, 0], [0, 0, dt], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    H = np.array([[-1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]])

    # KF covariances for 3D
    Q_kf = np.eye(9) * process_noise
    Q_kf[3, 3] = 2 * simulator.D * dt
    Q_kf[6, 6] = 2 * simulator.D * dt
    R_kf = np.eye(3) * measurement_noise  # 3D measurements

    # Instantiate controllers
    controllers = [LQGController(Ao, Bo, H, Q_kf, R_kf,
                                  lambda_x=lambda_x, lambda_u=lambda_u)
                   for _ in range(N)]

    # Initial loop
    for i, ctrl in enumerate(controllers):
        results['true_particles'][i, 0] = true_traj[i, 0]
        results['true_stage'][i, 0] = stage_pos[i]
        meas = true_traj[i, 0, :3] - stage_pos[i] + np.random.normal(0, measurement_noise, 3)
        results['measured'][i, 0] = meas + stage_pos[i]
        ctrl.x_hat = np.zeros(9)
        ctrl.update(meas)
        est = ctrl.x_hat
        results['est_particles'][i, 0] = est[:3]
        results['est_stage'][i, 0] = est[3:6]

    # Time loop
    for t in range(1, T):
        for i, ctrl in enumerate(controllers):
            # KF predict
            ctrl.predict(results['control_inputs'][i, t-1])
            # Measure
            true_rel = true_traj[i, t, :3] - stage_pos[i]
            meas = true_rel + np.random.normal(0, measurement_noise, 3)
            # KF update
            ctrl.update(meas)
            # Control
            u = ctrl.control()
            results['control_inputs'][i, t] = u

            # Apply to stage
            stage_pos[i] += u * dt
            results['true_stage'][i, t] = stage_pos[i]
            results['true_particles'][i, t] = true_traj[i, t]
            results['measured'][i, t] = meas + stage_pos[i]
            est = ctrl.x_hat
            results['est_particles'][i, t] = est[:3]
            results['est_stage'][i, t] = est[3:6]

    return results


# ===== Test Script =====
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initialize simulator
    sim = BrownianParticleSimulator3D(
        num_particles=1,
        duration=5,
        fps=20,
        temperature=300,
        viscosity=0.001,
        particle_radius=0.1e-6,
        bounds=[0, 20, 0, 20, 0, 20],  # 3D bounds
        drift=[0.1, 0.05, 0.01]  # 3D drift
    )

    # Run LQG tracking
    results = track_particles_lqg(
        sim,
        lambda_x=16.0,
        lambda_u=0.2,
        process_noise=0.1,
        measurement_noise=0.5
    )

    # Plot 3D trajectories for each particle
    N = results['true_particles'].shape[0]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        # True particle path
        ax.plot(results['true_particles'][i, :, 0],
                results['true_particles'][i, :, 1],
                results['true_particles'][i, :, 2],
                label=f'Particle {i + 1} True')
        # Stage tracking path
        ax.plot(results['true_stage'][i, :, 0],
                results['true_stage'][i, :, 1],
                results['true_stage'][i, :, 2],
                '--', label=f'Particle {i + 1} Stage')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (μm)')
    ax.set_title('True Particle Trajectories and Stage Tracking Paths in 3D')
    ax.legend()
    plt.show()

    # Compute and print RMSE
    errors = results['est_particles'] - results['true_particles']
    rmse = np.sqrt(np.mean(np.sum(errors ** 2, axis=2), axis=1))
    for i, r in enumerate(rmse, 1):
        print(f"Particle {i} RMSE: {r:.3f} μm")
