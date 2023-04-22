import distribution as dist
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, P, X, R = dist.normal_distribution(0,1), Q = dist.normal_distribution(0, 1)):
        self.Q = Q
        self.R = R
        self.P = P
        self.X = X

    def predict(self, F):
        self.X = F
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, H, Z):
        Y = Z - np.dot(H, self.X)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, Y)
        self.P = self.P - np.dot(np.dot(K, H), self.P)