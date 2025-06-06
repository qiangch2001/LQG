import numpy as np

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


def fluoro_bancroft(x_positions, y_positions, intensity_values, sigma_x, sigma_y, background):
    # Remove background intensity
    valid_intensity = np.maximum(intensity_values - background, 1e-6)
    
    # Calculate P2 parameter
    P2 = 2 * sigma_x ** 2 * np.log(valid_intensity)
    
    # Calculate alpha values
    alpha = 0.5 * (x_positions ** 2 + y_positions ** 2 + P2)
    
    # Form system of equations
    B = np.column_stack((x_positions, y_positions, np.ones_like(x_positions)))
    
    # Find pseudo-inverse
    B_pseudo_inv = np.linalg.pinv(B)
    
    # Solve for position
    pos = B_pseudo_inv @ alpha
    
    # Correct for non-isotropic PSF if necessary
    Q = np.array([[1, 0], [0, sigma_x / sigma_y]])
    estimated_position = Q @ pos[:2]
    
    return estimated_position[0], estimated_position[1]


def track_particles_lqg_with_fb(simulator, sampling_radius=10, num_samples=5, 
                                lambda_x=1.0, lambda_u=0.1, 
                                process_noise=0.1, measurement_noise=0.5,
                                sigma_x=0.203, sigma_y=0.203):
    
    true_traj = simulator.simulate()
    N, T, _ = true_traj.shape

    # Storage
    results = {key: np.zeros_like(true_traj) for key in [
        'true_particles', 'true_stage',
        'measured', 'est_particles', 'est_stage'
    ]}
    results['control_inputs'] = np.zeros((N, T, 2))

    stage_pos = np.zeros((N, 2))
    dt = simulator.dt

    # System matrices
    Ao = np.array([
        [1,0,0,0,0,0],
        [0,1,dt,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,dt],
        [0,0,0,0,0,1]
    ])
    Bo = np.array([[dt,0],[0,0],[0,0],[0,dt],[0,0],[0,0]])
    H = np.array([[-1,1,0,0,0,0],[0,0,0,-1,1,0]])

    # KF covariances
    Q_kf = np.eye(6) * process_noise
    Q_kf[2,2] = 2 * simulator.D * dt
    Q_kf[5,5] = 2 * simulator.D * dt
    R_kf = np.eye(2) * measurement_noise

    # Instantiate controllers
    controllers = [LQGController(Ao, Bo, H, Q_kf, R_kf,
                                  lambda_x=lambda_x, lambda_u=lambda_u)
                   for _ in range(N)]

    real_positions = []
    estimated_positions = []

    # Initial loop
    for i, ctrl in enumerate(controllers):
        results['true_particles'][i,0] = true_traj[i,0]
        results['true_stage'][i,0] = stage_pos[i]
        meas = true_traj[i,0] - stage_pos[i] + np.random.normal(0, measurement_noise, 2)
        results['measured'][i,0] = meas + stage_pos[i]
        ctrl.x_hat = np.zeros(6)
        ctrl.update(meas)
        est = ctrl.x_hat
        results['est_particles'][i,0] = [est[1], est[4]]
        results['est_stage'][i,0] = [est[0], est[3]]

    # Time loop
    for t in range(1, T):
        for i, ctrl in enumerate(controllers):
            # KF predict step
            ctrl.predict(results['control_inputs'][i,t-1])
            
            # Generate sampling points around predicted position
            true_particle_pos = true_traj[i,t]
            current_stage_pos = stage_pos[i]
            
            # Generate circle of sampling points
            angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
            x_samples = current_stage_pos[0] + sampling_radius * np.cos(angles)
            y_samples = current_stage_pos[1] + sampling_radius * np.sin(angles)
            
            # Calculate intensities (simulated)
            dx = x_samples - true_particle_pos[0]
            dy = y_samples - true_particle_pos[1]
            spatial_factors = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_x ** 2))
            signal = spatial_factors * 100  # Scale factor
            background = 10  # Background level
            intensity_values = signal + background + np.random.normal(0, measurement_noise, num_samples)
            
            # Apply FB method
            try:
                x_fb, y_fb = fluoro_bancroft(x_samples, y_samples, intensity_values, 
                                           sigma_x, sigma_y, background)
                # Convert to relative coordinates
                meas = np.array([x_fb - current_stage_pos[0], y_fb - current_stage_pos[1]])
            except:
                # Fallback to direct measurement with noise
                true_rel = true_particle_pos - current_stage_pos
                meas = true_rel + np.random.normal(0, measurement_noise, 2)

            # KF update
            ctrl.update(meas)
            
            # Control
            u = ctrl.control()
            results['control_inputs'][i,t] = u

            # Apply to stage - FIX: Apply each component separately
            stage_pos[i][0] += u[0] * dt
            stage_pos[i][1] += u[1] * dt
            
            # Store results
            results['true_stage'][i,t] = stage_pos[i]
            results['true_particles'][i,t] = true_traj[i,t]
            results['measured'][i,t] = np.array([x_fb, y_fb]) if 'x_fb' in locals() else meas + stage_pos[i]

            est = ctrl.x_hat
            results['est_particles'][i,t] = [est[1], est[4]]
            results['est_stage'][i,t] = [est[0], est[3]]

            current_true_position = true_traj[i,t]
            current_estimate = np.array([est[1], est[4]])

            real_positions.append(current_true_position)
            estimated_positions.append(current_estimate)

    real_positions = np.array(real_positions)
    estimated_positions = np.array(estimated_positions)


    # Plot real vs estimated trajectories
    plt.figure()
    plt.plot(real_positions[:, 0], real_positions[:, 1], label='True Particle Trajectory')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], '--', label='Estimated Trajectory')
    plt.xlabel('X position (μm)')
    plt.ylabel('Y position (μm)')
    plt.legend()
    plt.title('2D FB-LQG Particle Tracking')
    plt.grid()
    plt.axis('equal')  # Make X and Y axis same scale
    plt.show()

    return results


# ===== Test Script =====
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datagenerator import BrownianParticleSimulator

    # Initialize simulator
    sim = BrownianParticleSimulator(
        num_particles=3,
        duration=5,
        fps=20,
        temperature=300,
        viscosity=0.001,
        particle_radius=0.1e-6,
        bounds=[0, 50, 0, 50],
        drift=[0.1, 0.05]
    )

    # Run LQG tracking
    results = track_particles_lqg_with_fb(
        sim,
        sampling_radius=5,  # Sampling radius in μm
        num_samples=5,      # Number of sampling points
        lambda_x=1.0,
        lambda_u=0.1,
        process_noise=0.1,
        measurement_noise=0.5
    )

    # Plot trajectories for each particle
    N = results['true_particles'].shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(N):
        # True particle path
        plt.plot(results['true_particles'][i,:,0],
                 results['true_particles'][i,:,1],
                 label=f'Particle {i+1} True')
        # Stage tracking path
        plt.plot(results['true_stage'][i,:,0],
                 results['true_stage'][i,:,1],
                 '--', label=f'Particle {i+1} Stage')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title('True Particle Trajectories and Stage Tracking Paths')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute and print RMSE
    errors = results['est_particles'] - results['true_particles']
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=2), axis=1))
    for i, r in enumerate(rmse, 1):
        print(f"Particle {i} RMSE: {r:.3f} μm")

