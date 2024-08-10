import numpy as np

import DoublePendulum as dp

class ParticleFilter:
    def __init__(self, m1, m2, l1, l2, num_particles, state_dim, resampling_threshold=0.5, g = 9.81 ):

        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        self.resampling_threshold = resampling_threshold


    def initialize(self, initial_state_mean, initial_state_cov):
        self.particles = np.random.multivariate_normal(initial_state_mean, initial_state_cov, self.num_particles)
        self.weights.fill(1.0 / self.num_particles)


    def predict(self, control_input, motion_noise_cov):

        for i in range(self.num_particles):
            self.particles[i] = self.motion_model(self.particles[i], control_input) + np.random.multivariate_normal(np.zeros(self.state_dim), motion_noise_cov)


    def update(self, measurement, measurement_noise_cov):
        for i in range(self.num_particles):
            predicted_measurement = self.measurement_model(self.particles[i])
            self.weights[i] = self.gaussian_likelihood(measurement, predicted_measurement, measurement_noise_cov)
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)  # Normalize weights


    def resample(self):

        effective_num_particles = 1.0 / np.sum(np.square(self.weights))

        if effective_num_particles < self.resampling_threshold * self.num_particles:

            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)


    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)
    

    def motion_model(self, state, dt):

        theta1dotdot, theta2dotdot = dp.thetadotdot(self.m1, self.l1, self.m2, self.l2, state[0], state[1], state[2], state[3], self.g)
        
        theta1 = state[0] + dt * state[1] + 0.5 * (dt ** 2) * theta1dotdot
        theta2 = state[2] + dt * state[3] + 0.5 * (dt ** 2) * theta2dotdot
        theta1dot = state[1] + dt * theta1dotdot
        theta2dot = state[3] + dt * theta2dotdot

        return np.array([theta1, theta1dot, theta2, theta2dot])


    @staticmethod
    def gaussian_likelihood(measurement, predicted_measurement, measurement_noise_cov):
        error = measurement - predicted_measurement
        return np.exp(-0.5 * error.T @ np.linalg.inv(measurement_noise_cov) @ error) / np.sqrt((2 * np.pi)**len(measurement) * np.linalg.det(measurement_noise_cov))
