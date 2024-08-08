
import numpy as np

import DoublePendulum as dp

class UnscentedKalmanFilter:
    def __init__(self, m1, m2, l1, l2, P, X, R, Q, g = 9.81, alpha = 1e-3, beta = 2, kappa = 0):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        self.timeUpdated = 0.0

        self.Q = Q
        self.R = R
        self.P = P
        self.X = X
        self.N = len(X)
        self.g = g
        self.lmbda = alpha**2 * (self.N + kappa) - self.N

        self.weights_mean = np.zeros(self.N + 1)
        self.weights_cov  = np.zeros(self.N + 1)

        self.weights_mean[0] = self.lmbda / (self.lmbda + self.N)
        self.weights_cov[0]  = self.lmbda / (self.lmbda + self.N) + (1 - alpha**2 + beta)

        for i in range(self.N):
            self.weights_mean[i + 1] = 1.0 / (2.0 * (self.lmbda + self.N))
            self.weights_cov[i + 1] = 1.0 / (2.0 * (self.lmbda + self.N))


        self.sensor = 0

        self.timeHist   = [0.0]
        self.theta1Hist = [self.X[0]]
        self.theta2Hist = [self.X[2]]


    def _constructSigmaPoints(self, x, P) -> np.ndarray:

        Xsig = np.zeros((self.N, 2 * self.N + 1))

        scaledP = np.linalg.cholesky((self.lmbda + self.N) * P)

        Xsig[:,0] = x

        for i in range(self.N):
            Xsig[:,i + 1] = x + scaledP[i,:]
            Xsig[:,i + self.N + 1] = x - scaledP[i,:]


        return Xsig


    def _propagateSigmaPoints(self, Xsig, dt) -> np.ndarray:
        
        return np.transpose(np.array([self._nonlinearFunction(Xsig[:,i], dt) for i in range(2 * self.N + 1)]))


    def _nonlinearFunction(self, stateVec, dt) -> np.ndarray:
        theta1dotdot, theta2dotdot = dp.thetadotdot(self.m1, self.l1, self.m2, self.l2, stateVec[0], stateVec[1], stateVec[2], stateVec[3], self.g)

        theta1      = stateVec[0] + dt * stateVec[1] + 0.5 * (dt ** 2) * theta1dotdot
        theta1dot   = stateVec[1] + dt * theta1dotdot
        theta2      = stateVec[2] + dt * stateVec[3] + 0.5 * (dt ** 2) * theta2dotdot
        theta2dot   = stateVec[3] + dt * theta2dotdot

        return np.array([theta1, theta1dot, theta2, theta2dot])


    def _predictMean(self, Xsig_prop) -> np.ndarray:

        Xpred = self.weights_mean[0] * Xsig_prop[:, 0]

        for i in range(self.N):
            Xpred += self.weights_mean[i + 1] * Xsig_prop[:, i + 1]
            Xpred += self.weights_mean[i + 1] * Xsig_prop[:, i + 1 + self.N]

        return Xpred


    def _predictCovariance(self, Xsig_prop, Xpred) -> np.ndarray:
        Ppred = self.Q.copy()

        Ppred += self.weights_cov[0] * np.outer(Xsig_prop[:, 0] - Xpred, Xsig_prop[:, 0] - Xpred)

        for i in range(self.N):
            Ppred += self.weights_cov[i + 1] * np.outer(Xsig_prop[:, i + 1] - Xpred, Xsig_prop[:, i + 1] - Xpred)
            Ppred += self.weights_cov[i + 1] * np.outer(Xsig_prop[:, i + 1 + self.N] - Xpred, Xsig_prop[:, i + 1 + self.N] - Xpred)


        return Ppred


    def _predictedMeasurementCovariance(self, Zsig, Zpred) -> np.ndarray:

        S = self.R.copy()

        S += self.weights_cov[0] * np.outer(Zsig[:, 0] - Zpred, Zsig[:, 0] - Zpred)

        for i in range(self.N):
            S += self.weights_cov[i + 1] * np.outer(Zsig[:, i + 1] - Zpred, Zsig[:, i + 1] - Zpred)
            S += self.weights_cov[i + 1] * np.outer(Zsig[:, i + 1 + self.N] - Zpred, Zsig[:, i + 1 + self.N] - Zpred)

        return S


    def _crossCorrelation(self, Xsig_prop, Xpred, Zsig, Zpred) -> np.ndarray:
        T = self.weights_cov[0] * np.outer(Xsig_prop[:, 0] - Xpred, Zsig[:, 0] - Zpred)

        for i in range(self.N):
            T += self.weights_cov[i + 1] * np.outer(Xsig_prop[:, i + 1] - Xpred, Zsig[:, i + 1] - Zpred)
            T += self.weights_cov[i + 1] * np.outer(Xsig_prop[:, i + 1 + self.N] - Xpred, Zsig[:, i + 1 + self.N] - Zpred)

        return T


    def _constructSigmaPointsInMeasurementSpace(self, Xsig_prop) -> np.ndarray:
        if self.sensor == 1:
            Zsig = Xsig_prop[0, :].copy()

        elif self.sensor == 2:
            Zsig = Xsig_prop[2, :].copy()

        else:
            Zsig = np.zeros((2, (2 * self.N + 1)))

            Zsig[0, :] = Xsig_prop[0, :].copy()
            Zsig[1, :] = Xsig_prop[2, :].copy()

        return Zsig


    def _predictionInMeasurementSpace(self, Zsig) -> np.ndarray:

        if self.sensor != 0:
            
            Zpred = self.weights_mean[0] * Zsig[0]

            for i in range(self.N):
                Zpred += self.weights_mean[i + 1] * Zsig[i + 1]
                Zpred += self.weights_mean[i + 1] * Zsig[i + 1 + self.N]

        else:
            Zpred = self.weights_mean[0] * Zsig[:, 0]

            for i in range(self.N):
                Zpred += self.weights_mean[i + 1] * Zsig[:, i + 1]
                Zpred += self.weights_mean[i + 1] * Zsig[:, i + 1 + self.N]

        return Zpred


    def predict(self, t) -> None:

        Xsig = self._constructSigmaPoints(self.X, self.P)
        Xsig_prop = self._propagateSigmaPoints(Xsig, t - self.timeUpdated)
        Xpred = self._predictMean(Xsig_prop)
        Ppred = self._predictCovariance(Xsig_prop, Xpred)

        return Xpred, Ppred, Xsig_prop


    def _update(self, t, Y) -> None:

        Xpred, Ppred, Xsig_prop = self.predict(t)

        Zsig = self._constructSigmaPointsInMeasurementSpace(Xsig_prop)
        Zpred = self._predictionInMeasurementSpace(Zsig)

        dY = Y - Zpred

        S = self._predictedMeasurementCovariance(Zsig, Zpred)
        T = self._crossCorrelation(Xsig_prop, Xpred, Zsig, Zpred)
        K = T @ np.linalg.inv(S)


        self.X = Xpred + K @ dY
        
        self.P = Ppred - K @ S @ np.transpose(K)


    def newData(self, z, t, sensor = 0) -> None:
        self.sensor = sensor

        self._update(t, z)

        self.timeUpdated = t


        self.timeHist.append(t)
        self.theta1Hist.append(self.X[0])
        self.theta2Hist.append(self.X[2])
    

    def _wrapAngle(self, angle) -> np.float64:

        ratio = np.round(angle / (2.0 * np.pi))

        angle -= 2.0 * np.pi * ratio

        if angle >= np.pi:
            angle -= 2.0 * np.pi

        elif angle < -np.pi:
            angle += 2.0 * np.pi

        return angle