import numpy as np
import matplotlib.pyplot as plt
from datagenerator import BrownianParticleSimulator

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


def generate_circle_points(center, radius, num_points): 
    """Generate measurement points in a circle around the center"""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_positions = center[0] + radius * np.cos(angles)
    y_positions = center[1] + radius * np.sin(angles)
    return x_positions, y_positions


def simulate_fluorescence(x_positions, y_positions, true_position, 
                         sigma_x, max_intensity, background_rate, time_per_point):
    """Simulate fluorescence intensity measurements"""
    dx = x_positions - true_position[0]
    dy = y_positions - true_position[1]
    spatial_factors = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_x ** 2))
    
    # Mean signal intensity
    signal_mean = max_intensity * time_per_point * spatial_factors
    
    # Add noise
    background_mean = background_rate * time_per_point
    signal_photons = np.random.poisson(signal_mean)
    background_photons = np.random.poisson(background_mean, size=len(x_positions))
    intensity_values = signal_photons + background_photons
    
    return intensity_values, background_mean


def fluoro_bancroft(x, y, intensity, sigma_x, sigma_y, background_intensity):
    """
    FluoroBancroft algorithm for position estimation from intensity measurements
    """
    valid_intensity = np.maximum(intensity - background_intensity, 1e-6)
    P2 = 2 * sigma_x ** 2 * np.log(valid_intensity)
    alpha = 0.5 * (x ** 2 + y ** 2 + P2)
    B = np.column_stack((x, y, np.ones_like(x)))
    
    try:
        B_pseudo_inv = np.linalg.pinv(B)
        pos = B_pseudo_inv @ alpha
        Q = np.array([[1, 0], [0, sigma_x / sigma_y]])
        estimated_position = Q @ pos[:2]
        return estimated_position[0], estimated_position[1]
    except:
        # Fallback to weighted average if matrix inversion fails
        total_intensity = np.sum(intensity)
        if total_intensity <= 0:
            return np.mean(x), np.mean(y)
        
        x0 = np.sum(x * intensity) / total_intensity
        y0 = np.sum(y * intensity) / total_intensity
        return x0, y0


def track_particles_lqg_fb(simulator, num_scan_points=5, scan_radius=0.5, 
                          psf_sigma=0.2, max_intensity=50000, background_rate=200,
                          time_per_point=0.0005, lambda_x=1.0, lambda_u=0.1,
                          process_noise=0.1, measurement_noise=0.5):
    """
    Track particles using LQG control with FluoroBancroft position estimation
    
    Parameters:
    -----------
    simulator : BrownianParticleSimulator
        Simulator object with particle trajectories
    num_scan_points : int
        Number of measurement points per scan
    scan_radius : float
        Radius of the circular scan pattern (μm)
    psf_sigma : float
        Width (sigma) of the Gaussian PSF (μm)
    max_intensity : float
        Maximum intensity at the center of the PSF (photons/second)
    background_rate : float
        Background photon rate (photons/second)
    time_per_point : float
        Measurement time per scan point (seconds)
    lambda_x, lambda_u : float
        LQR weights for state error and control effort
    process_noise, measurement_noise : float
        Noise parameters for Kalman filter
    
    Returns:
    --------
    results : dict
        Dictionary with tracking results
    """
    true_traj = simulator.simulate()
    N, T, _ = true_traj.shape

    # Storage for results
    results = {key: np.zeros_like(true_traj) for key in [
        'true_particles', 'true_stage',
        'fb_estimated', 'est_particles', 'est_stage'
    ]}
    results['control_inputs'] = np.zeros((N, T, 2))
    results['scan_centers'] = np.zeros((N, T, 2))

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

    # Initial loop
    for i, ctrl in enumerate(controllers):
        results['true_particles'][i,0] = true_traj[i,0]
        results['true_stage'][i,0] = stage_pos[i]
        results['scan_centers'][i,0] = stage_pos[i]
        
        # Initial scan points
        x_scan, y_scan = generate_circle_points(stage_pos[i], scan_radius, num_scan_points)
        
        # Simulate fluorescence measurements
        intensities, bg_mean = simulate_fluorescence(
            x_scan, y_scan, true_traj[i,0], 
            psf_sigma, max_intensity, background_rate, time_per_point
        )
        
        # Estimate position using FluoroBancroft
        try:
            x_fb, y_fb = fluoro_bancroft(x_scan, y_scan, intensities, psf_sigma, psf_sigma, bg_mean)
            fb_pos = np.array([x_fb, y_fb])
        except:
            # Fallback if FB fails
            fb_pos = stage_pos[i]
        
        results['fb_estimated'][i,0] = fb_pos
        
        # Calculate relative measurement for KF
        meas = fb_pos - stage_pos[i]
        
        # Initial state estimate
        ctrl.x_hat = np.zeros(6)
        ctrl.update(meas)
        est = ctrl.x_hat
        results['est_particles'][i,0] = [est[1], est[4]]
        results['est_stage'][i,0] = [est[0], est[3]]

    # Time loop
    for t in range(1, T):
        for i, ctrl in enumerate(controllers):
            # KF predict
            ctrl.predict(results['control_inputs'][i,t-1])
            
            # Predicted particle position for scan center
            pred_particle_pos = np.array([ctrl.x_hat[1], ctrl.x_hat[4]])
            scan_center = stage_pos[i]
            results['scan_centers'][i,t] = scan_center
            
            # Generate scan points around current stage position
            x_scan, y_scan = generate_circle_points(scan_center, scan_radius, num_scan_points)
            
            # Simulate fluorescence measurements
            intensities, bg_mean = simulate_fluorescence(
                x_scan, y_scan, true_traj[i,t], 
                psf_sigma, max_intensity, background_rate, time_per_point
            )
            
            # Estimate position using FluoroBancroft
            try:
                x_fb, y_fb = fluoro_bancroft(x_scan, y_scan, intensities, psf_sigma, psf_sigma, bg_mean)
                fb_pos = np.array([x_fb, y_fb])
            except:
                # Fallback if FB fails
                fb_pos = scan_center
            
            results['fb_estimated'][i,t] = fb_pos
            
            # Calculate relative measurement for KF
            meas = fb_pos - stage_pos[i]
            
            # KF update
            ctrl.update(meas)
            
            # Control
            u = ctrl.control()
            results['control_inputs'][i,t] = u

            # Apply to stage
            stage_pos[i] += u * dt
            
            # Store results
            results['true_stage'][i,t] = stage_pos[i]
            results['true_particles'][i,t] = true_traj[i,t]
            est = ctrl.x_hat
            results['est_particles'][i,t] = [est[1], est[4]]
            results['est_stage'][i,t] = [est[0], est[3]]

    return results


# ===== Visualization Functions =====
def plot_tracking_results(results, particle_indices=None):
    """Plot tracking results"""
    N = results['true_particles'].shape[0]
    if particle_indices is None:
        particle_indices = range(N)
    
    plt.figure(figsize=(12, 10))
    
    for i in particle_indices:
        # True particle trajectory
        plt.plot(results['true_particles'][i,:,0], 
                 results['true_particles'][i,:,1], 
                 '-', label=f'Particle {i+1} True')
        
        # Stage position (tracking)
        plt.plot(results['true_stage'][i,:,0], 
                 results['true_stage'][i,:,1], 
                 '--', label=f'Particle {i+1} Stage')
        
        # FB estimated positions
        plt.plot(results['fb_estimated'][i,:,0], 
                 results['fb_estimated'][i,:,1], 
                 '.', markersize=4, alpha=0.5, 
                 label=f'Particle {i+1} FB Estimate')
    
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title('Particle Tracking with FluoroBancroft Position Estimation')
    plt.legend()
    plt.grid(True)
    
    # Compute and print RMSE
    fb_errors = results['fb_estimated'] - results['true_particles']
    fb_rmse = np.sqrt(np.mean(np.sum(fb_errors**2, axis=2), axis=1))
    
    kf_errors = results['est_particles'] - results['true_particles']
    kf_rmse = np.sqrt(np.mean(np.sum(kf_errors**2, axis=2), axis=1))
    
    for i in particle_indices:
        print(f"Particle {i+1}:")
        print(f"  FB RMSE: {fb_rmse[i]:.3f} μm")
        print(f"  KF RMSE: {kf_rmse[i]:.3f} μm")
    
    plt.tight_layout()
    plt.show()


# ===== Test Script =====
if __name__ == "__main__":
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

    # Run LQG tracking with FluoroBancroft
    results = track_particles_lqg_fb(
        sim,
        num_scan_points=5,
        scan_radius=0.5,
        psf_sigma=0.2,
        max_intensity=50000,
        background_rate=200,
        time_per_point=0.0005,
        lambda_x=1.0,
        lambda_u=0.1,
        process_noise=0.1,
        measurement_noise=0.5
    )

    # Plot results
    plot_tracking_results(results)