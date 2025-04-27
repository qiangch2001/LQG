import numpy as np
import matplotlib.pyplot as plt
from datagenerator import BrownianParticleSimulator

class LQGController:
    def __init__(self, A, B, H, Q_kf, R_kf, lambda_x=1.0, lambda_u=0.1):
        self.A = A
        self.B = B
        self.H = H
        self.Q_kf = Q_kf
        self.R_kf = R_kf
        self.lambda_x = lambda_x
        self.lambda_u = lambda_u

        self.n = A.shape[0]
        self.m = B.shape[1]

        self.G = self._compute_lqr_gain()

        self.x_hat = np.zeros((self.n,))
        self.P = np.eye(self.n)

    def _compute_lqr_gain(self, max_iters=1000, tol=1e-6):
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


def simulate_fluorescence(xp, yp, meas_points, max_intensity=200000, beam_waist=0.5, background_rate=2000, dt=0.004):
    intensities = []
    for x, y in meas_points:
        signal = max_intensity * np.exp(-((x - xp)**2 + (y - yp)**2) / (beam_waist**2))
        signal_counts = signal * dt
        background_counts = background_rate * dt
        total_counts = signal_counts + background_counts
        detected_photons = np.random.poisson(total_counts)
        intensities.append(detected_photons)
    return np.array(intensities)


def estimate_position_fb(meas_points, intensities, beam_waist=0.5):
    weights = np.log(np.maximum(intensities, 1))  # avoid log(0)
    w_sum = np.sum(weights)
    x_est = np.sum(weights * np.array([p[0] for p in meas_points])) / w_sum
    y_est = np.sum(weights * np.array([p[1] for p in meas_points])) / w_sum
    return np.array([x_est, y_est])


def track_particles_lqg(simulator, lambda_x=1.0, lambda_u=0.1, process_noise=0.1, measurement_noise=0.5):
    true_traj = simulator.simulate()
    N, T, _ = true_traj.shape

    results = {key: np.zeros_like(true_traj) for key in [
        'true_particles', 'true_stage',
        'measured', 'est_particles', 'est_stage']}
    results['control_inputs'] = np.zeros((N, T, 2))

    stage_pos = np.zeros((N, 2))
    dt = simulator.dt

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

    Q_kf = np.eye(6) * process_noise
    Q_kf[2,2] = 2 * simulator.D * dt
    Q_kf[5,5] = 2 * simulator.D * dt
    R_kf = np.eye(2) * measurement_noise

    controllers = [LQGController(Ao, Bo, H, Q_kf, R_kf,
                                  lambda_x=lambda_x, lambda_u=lambda_u)
                   for _ in range(N)]

    # Parameters for FB
    num_points = 5
    radius = 0.5  # μm
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    for i, ctrl in enumerate(controllers):
        results['true_particles'][i,0] = true_traj[i,0]
        results['true_stage'][i,0] = stage_pos[i]

        meas_points = np.array([
            stage_pos[i] + np.array([np.cos(a), np.sin(a)]) * radius
            for a in angles
        ])
        intensities = simulate_fluorescence(true_traj[i,0,0], true_traj[i,0,1], meas_points, dt=dt)
        meas = estimate_position_fb(meas_points, intensities)

        results['measured'][i,0] = meas
        ctrl.x_hat = np.zeros(6)
        ctrl.update(meas)
        est = ctrl.x_hat
        results['est_particles'][i,0] = [est[1], est[4]]
        results['est_stage'][i,0] = [est[0], est[3]]

    for t in range(1, T):
        for i, ctrl in enumerate(controllers):
            ctrl.predict(results['control_inputs'][i,t-1])

            meas_points = np.array([
                stage_pos[i] + np.array([np.cos(a), np.sin(a)]) * radius
                for a in angles
            ])
            intensities = simulate_fluorescence(true_traj[i,t,0], true_traj[i,t,1], meas_points, dt=dt)
            meas = estimate_position_fb(meas_points, intensities)

            ctrl.update(meas)
            u = ctrl.control()
            results['control_inputs'][i,t] = u

            stage_pos[i] += u * dt
            results['true_stage'][i,t] = stage_pos[i]
            results['true_particles'][i,t] = true_traj[i,t]
            results['measured'][i,t] = meas
            est = ctrl.x_hat
            results['est_particles'][i,t] = [est[1], est[4]]
            results['est_stage'][i,t] = [est[0], est[3]]

    return results


if __name__ == "__main__":
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

    results = track_particles_lqg(
        sim,
        lambda_x=1.0,
        lambda_u=0.1,
        process_noise=0.1,
        measurement_noise=0.5
    )

    N = results['true_particles'].shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(N):
        plt.plot(results['true_particles'][i,:,0],
                 results['true_particles'][i,:,1],
                 label=f'Particle {i+1} True')
        plt.plot(results['true_stage'][i,:,0],
                 results['true_stage'][i,:,1],
                 '--', label=f'Particle {i+1} Stage')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title('True Particle Trajectories and Stage Tracking Paths')
    plt.legend()
    plt.grid(True)
    plt.show()

    errors = results['est_particles'] - results['true_particles']
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=2), axis=1))
    for i, r in enumerate(rmse, 1):
        print(f"Particle {i} RMSE: {r:.3f} μm")
