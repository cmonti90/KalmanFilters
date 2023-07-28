import distribution as dist
import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, m1, m2, l1, l2, P, X, R, Q, g = 9.81, lmbda = None):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        self.timeUpdated = 0.0
        self.Q = Q
        self.R = R
        self.P = P
        self.Ppred = P
        self.X = X
        self.Xpred = np.zeros((len(self.X),))
        self.N = len(self.X)
        self.Xsig = np.zeros((len(self.X), (2 * len(self.X) + 1)))
        self.Xsig_prop = np.zeros((len(self.X), (2 * len(self.X) + 1)))
        self.Zhat = np.zeros((2,))
        self.Z = np.zeros((2,))
        self.Zsig = np.zeros((2, (2 * len(self.X) + 1)))
        self.g = g
        self.lmbda = lmbda

        if self.lmbda is None:
            self.lmbda = 3 - self.N    

        self._defineWeights()

        self.theta1Hist = [self.X[0]]
        self.theta2Hist = [self.X[2]]

    def _defineWeights(self) -> None:
        self.w = np.zeros((2 * self.N + 1),)
        self.w[0] = self.lmbda / (self.lmbda + self.N)

        for i in range(2 * self.N + 1):
            self.w[i] = 1.0 / (2.0 * (self.lmbda + self.N))


    def _constructSigmaPoints(self, x, P) -> None:
        self.Xsig[:, 0] = x

        scaledP = np.sqrt((self.lmbda + self.N) * P)

        for i in range(self.N):
            self.Xsig[:, i] = x + scaledP[:, i]
            self.Xsig[:, i + self.N] = x - scaledP[:, i]


    def _propagateSigmaPoints(self, dt) -> None:
        self.Xsig_prop[:, 0] = self._nonlinearFunction(self.Xsig[:, 0], dt)

        for i in range(2 * self.N):
            j = i + 1
            sigmaIdx = self.N - (j % self.N)
            self.Xsig_prop[:, sigmaIdx] = self._nonlinearFunction(self.Xsig[:, sigmaIdx], dt)


    def _nonlinearFunction(self, stateVec, dt) -> np.ndarray:
        dTheta = stateVec[0] - stateVec[2]
        M = self.m1 + self.m2
        theta1dotdot = (-np.sin(dTheta) * self.m2 * (self.l1 * (stateVec[1] ** 2) * np.cos(dTheta) + self.l2 * (stateVec[2] ** 2)) - self.g * (M * np.sin(stateVec[0]) - self.m2 * np.sin(stateVec[2]) * np.cos(dTheta))) / (self.l1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
        theta2dotdot = (np.sin(dTheta) * (M * self.l1 * (stateVec[1] ** 2) + self.l2 * self.m2 * (stateVec[3] ** 2) * np.cos(dTheta)) + self.g * M * (np.sin(stateVec[0]) * np.cos(dTheta) - np.sin(stateVec[2]))) / (self.l2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))

        state1 = stateVec[0] + dt * stateVec[1] + 0.5 * (dt ** 2) * theta1dotdot
        state2 = stateVec[1] + dt * theta1dotdot
        state3 = stateVec[2] + dt * stateVec[3] + 0.5 * (dt ** 2) * theta2dotdot
        state4 = stateVec[3] + dt * theta2dotdot

        return np.array([state1, state2, state3, state4])


    def _predictMean(self) -> None:
        self.Xpred = self.w[0] * self.Xsig_prop[:, 0]

        for i in range(2 * self.N):
            j = i + 1
            sigmaIdx = self.N - (j % self.N)
            self.Xpred += self.w[sigmaIdx] * self.Xsig_prop[:, sigmaIdx]


    def _predictCovariance(self) -> None:
        self.Ppred = self.w[0] * np.outer(self.Xsig_prop[:, 0] - self.Xpred, self.Xsig_prop[:, 0] - self.Xpred)

        for i in range(2 * self.N):
            j = i + 1
            sigmaIdx = self.N - (j % self.N)
            self.Ppred += self.w[sigmaIdx] * np.outer(self.Xsig_prop[:, sigmaIdx] - self.Xpred, self.Xsig_prop[:, sigmaIdx] - self.Xpred)

        self.Ppred += self.Q


    def _computePredictedMeasurementCovariance(self) -> np.ndarray:
        S = self.w[0] * np.outer(self.Zsig[:, 0] - self.Zhat, self.Zsig[:, 0] - self.Zhat)

        for i in range(2 * self.N):
            j = i + 1
            sigmaIdx = self.N - (j % self.N)
            S += self.w[sigmaIdx] * np.outer(self.Zsig[:, sigmaIdx] - self.Zhat, self.Zsig[:, sigmaIdx] - self.Zhat)

        S += self.R

        return S


    def _computeCrossCorrelation(self) -> np.ndarray:
        T = self.w[0] * np.outer(self.Xsig_prop[:, 0] - self.Xpred, self.Zsig[:, 0] - self.Zhat)

        for i in range(2 * self.N):
            j = i + 1
            sigmaIdx = self.N - (j % self.N)
            T += self.w[sigmaIdx] * np.outer(self.Xsig_prop[:, sigmaIdx] - self.Xpred, self.Zsig[:, sigmaIdx] - self.Zhat)

        return T


    def _constructSigmaPointsInMeasurementSpace(self, dt) -> None:
        for i in range(2 * self.N + 1):
            self.Zsig[0, i] = self.Xsig_prop[0, i]
            self.Zsig[1, i] = self.Xsig_prop[2, i]


    def _predictionInMeasurementSpace(self, dt) -> None:
        self._constructSigmaPointsInMeasurementSpace(dt)
        
        self.Zhat = self.w[0] * self.Zsig[:, 0]

        for i in range(2 * self.N):
            j = i + 1
            sigmaIdx = self.N - (j % self.N)
            self.Zhat += self.w[sigmaIdx] * self.Zsig[:, sigmaIdx]


    def predict(self, t) -> None:
        self._constructSigmaPoints(self.X, self.P)
        self._propagateSigmaPoints(t - self.timeUpdated)
        self._predictMean()
        self._predictCovariance()


    def _update(self, Y, t) -> None:
        # self._constructSigmaPoints(self.Xpred, self.Ppred)
        self._predictionInMeasurementSpace(t - self.timeUpdated)

        dY = Y - self.Zhat
        S = self._computePredictedMeasurementCovariance()
        T = self._computeCrossCorrelation()
        K = T @ np.linalg.inv(S)

        self.X = self.Xpred + K @ dY
        
        self.P = self.Ppred - K @ np.transpose(T)
        # self.P = self.Ppred - K @ S @ np.transpose(K)


    def newData(self, z, t) -> None:
        self.predict(t)
        self._update(z, t)
        self.timeUpdated = t
        self.theta1Hist.append(self.X[0])
        self.theta2Hist.append(self.X[2])