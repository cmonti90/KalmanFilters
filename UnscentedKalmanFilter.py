
import numpy as np

import DoublePendulum as dp

class UnscentedKalmanFilter:
    def __init__(self, m1, m2, l1, l2, P0, X0, R, Q, g = 9.81, alpha = 1e-3, beta = 2, kappa = 0):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        self.timeUpdated = 0.0

        self.Q = Q
        self.R = R.copy()
        self.P = P0.copy()
        self.X = X0.copy()
        self.N = len(self.X)
        self.g = g
        self.lmbda = alpha**2 * (self.N + kappa) - self.N

        self.weights_mean = np.zeros(self.N + 1)
        self.weights_cov  = np.zeros(self.N + 1)

        self.weights_mean[0] = self.lmbda / (self.lmbda + self.N)
        self.weights_cov [0] = self.lmbda / (self.lmbda + self.N) + (1 - alpha**2 + beta)

        for i in range(self.N):
            self.weights_mean[i + 1] = 1.0 / (2.0 * (self.lmbda + self.N))
            self.weights_cov [i + 1] = 1.0 / (2.0 * (self.lmbda + self.N))

        self.sensorIdx = 0
        self.measStateIdx = (0,2)


    def _constructSigmaPoints(self):

        Xsig = np.zeros((self.N, 2 * self.N + 1))
        
        scaleFactor = np.sqrt(self.lmbda + self.N)

        sqrtScaledP = scaleFactor * np.linalg.cholesky(self.P)

        Xsig[:,(0,)] = self.X.copy()

        for i in range(self.N):
            idx1 = i + 1
            idx2 = idx1 + self.N
            Xsig[:,(idx1,)] = self.X + sqrtScaledP[:,(i,)]
            Xsig[:,(idx2,)] = self.X - sqrtScaledP[:,(i,)]

            # Xsig[(0,2),(idx1,idx2)] = dp.wrapNegPiToPi(Xsig[(0,2),(idx1,idx2)])


        return Xsig


    def _nonlinearFunction(self, stateVec, dt):
        theta1dotdot, theta2dotdot = dp.thetadotdot(self.m1, self.l1, self.m2, self.l2, stateVec[0], stateVec[1], stateVec[2], stateVec[3], self.g)

        theta1      = stateVec[0] + dt * stateVec[1] + 0.5 * (dt ** 2) * theta1dotdot
        theta1dot   = stateVec[1] + dt * theta1dotdot
        theta2      = stateVec[2] + dt * stateVec[3] + 0.5 * (dt ** 2) * theta2dotdot
        theta2dot   = stateVec[3] + dt * theta2dotdot

        theta1 = dp.wrapNegPiToPi(theta1)
        theta2 = dp.wrapNegPiToPi(theta2)

        return np.array([theta1, theta1dot, theta2, theta2dot])


    def _predictMean(self, Xsig_prop):

        Xpred = self.weights_mean[0] * Xsig_prop[:, (0,)]

        for i in range(self.N):
            idx1 = i + 1
            idx2 = idx1 + self.N
            Xpred += self.weights_mean[idx1] * ( Xsig_prop[:, (idx1,)] + Xsig_prop[:, (idx2,)] )

        Xpred[(0,2),] = dp.wrapNegPiToPi(Xpred[(0,2),])

        return Xpred


    def _predictCovariance(self, Xsig_prop, Xpred, dt):
        Ppred = self.Q(dt)

        Xsig_prop_wrap = Xsig_prop.copy()
        Xsig_prop_wrap[(0,2),:] = dp.wrapNegPiToPi(Xsig_prop_wrap[(0,2),:])

        dX = Xsig_prop_wrap - Xpred
        dX[(0,2),:] = dp.findMinAngleDifference( Xsig_prop_wrap[(0,2),:], Xpred[(0,2),] )

        Ppred += self.weights_cov[0] * np.outer(dX[:, (0,)], dX[:, (0,)])

        for i in range(self.N):
            idx1 = i + 1
            idx2 = idx1 + self.N

            Ppred += self.weights_cov[idx1] * np.outer(dX[:, (idx1,)], dX[:, (idx1,)])
            Ppred += self.weights_cov[idx1] * np.outer(dX[:, (idx2,)], dX[:, (idx2,)])


        return Ppred


    def _predictedMeasurementCovariance(self, Zsig_prop, Zpred):

        if self.sensorIdx == 0:
            S = self.R.copy()

        elif self.sensorIdx == 1:
            S = self.R[0,0].copy()

        elif self.sensorIdx == 2:
            S = self.R[1,1].copy()

        else:
            raise ValueError("Invalid sensorIdx value")


        dZ = dp.findMinAngleDifference( Zsig_prop, Zpred )
    
        S += self.weights_cov[0] * np.outer(dZ[:, (0,)], dZ[:, (0,)])

        for i in range(self.N):
            idx1 = i + 1
            idx2 = idx1 + self.N
            S += self.weights_cov[i + 1] * np.outer(dZ[:, (idx1,)], dZ[:, (idx1,)])
            S += self.weights_cov[i + 1] * np.outer(dZ[:, (idx2,)], dZ[:, (idx2,)])

        return S


    def _crossCorrelation(self, Xsig_prop, Xpred, Zsig_prop, Zpred):

        Xsig_prop_wrap = Xsig_prop.copy()
        Xsig_prop_wrap[(0,2),:] = dp.wrapNegPiToPi(Xsig_prop_wrap[(0,2),:])

        dX = Xsig_prop_wrap - Xpred
        dX[(0,2),:] = dp.findMinAngleDifference( Xsig_prop_wrap[(0,2),:], Xpred[(0,2),] )

        dZ = dp.findMinAngleDifference( Zsig_prop, Zpred )

        T = self.weights_cov[0] * np.outer(dX[:, (0,)], dZ[:, (0,)])

        for i in range(self.N):
            idx1 = i + 1
            idx2 = idx1 + self.N
            T += self.weights_cov[idx1] * np.outer(dX[:, (idx1,)], dZ[:, (idx1,)])
            T += self.weights_cov[idx1] * np.outer(dX[:, (idx2,)], dZ[:, (idx2,)])

        return T


    def _predictionInMeasurementSpace(self, Zsig_prop):

        Zpred = self.weights_mean[0] * Zsig_prop[:, (0,)]

        for i in range(self.N):
            idx1 = i + 1
            idx2 = idx1 + self.N
            Zpred += self.weights_mean[idx1] * Zsig_prop[:, (idx1,)]
            Zpred += self.weights_mean[idx1] * Zsig_prop[:, (idx2,)]

        Zpred = dp.wrapNegPiToPi(Zpred)


        return Zpred


    def predict(self, t):

        dt = t - self.timeUpdated

        Xsig = self._constructSigmaPoints()

        Xsig_prop = np.transpose(np.array([self._nonlinearFunction(Xsig[:,i], dt) for i in range(2 * self.N + 1)]))

        Xpred = self._predictMean(Xsig_prop)
        Ppred = self._predictCovariance(Xsig_prop, Xpred, dt)

        return Xpred, Ppred, Xsig_prop


    def update(self, t, Y, sensorIdx):

        self._determineMeasurementIndexing(sensorIdx)

        Xpred, Ppred, Xsig_prop = self.predict(t)

        Zsig_prop = Xsig_prop[self.measStateIdx, :].copy()
        Zpred = self._predictionInMeasurementSpace(Zsig_prop)

        dY = dp.findMinAngleDifference(Y, Zpred)

        S = self._predictedMeasurementCovariance(Zsig_prop, Zpred)
        T = self._crossCorrelation(Xsig_prop, Xpred, Zsig_prop, Zpred)

        K = T @ np.linalg.inv(S)

        self.X = Xpred + K @ dY

        self.X[0] = dp.wrapNegPiToPi(self.X[0])
        self.X[2] = dp.wrapNegPiToPi(self.X[2])
        
        self.P = Ppred - K @ S @ K.T

        self.timeUpdated = t

        return self.X.copy()
    

    def _determineMeasurementIndexing(self, sensorIdx):

        self.sensorIdx = sensorIdx

        if sensorIdx == 1:
            self.measStateIdx = (0,)

        elif sensorIdx == 2:
            self.measStateIdx = (2,)

        elif sensorIdx == 0:
            self.measStateIdx = (0, 2)

        else:
            raise ValueError("Invalid sensorIdx value")