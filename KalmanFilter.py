import distribution as dist
import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, P, X, R = dist.normal_distribution(0,1), Q = dist.normal_distribution(0, 1)):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.X = X

    def predict(self, U):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, U)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, Z):
        Y = Z - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, Y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)