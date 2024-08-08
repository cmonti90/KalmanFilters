import numpy as np

import DoublePendulum as dp

class ExtendedKalmanFilter:
    def __init__(self, m1, m2, L1, L2, P, X, R, Q, g = 9.81):
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2

        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.X = X.copy()
        
        self.timeUpdated = 0.0
        self.g = g

        self.timeHist   = [0.0]
        self.theta1Hist = [X[0]]
        self.theta2Hist = [X[2]]


    def _f11(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = -np.cos(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) + (np.sin(dTheta) ** 2) * (self.m2 * self.L1 * (self.X[1] ** 2)) - self.g * (M * np.cos(self.X[0]) + self.m2 * np.sin(self.X[2]) * np.sin(dTheta))
        term3 = 2.0 * self.L1 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = -np.sin(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - self.g * (M * np.sin(self.X[0]) - self.m2 * np.sin(self.X[2]) * np.cos(dTheta))

        denom = term1 ** 2

        return 1.0 + ((dt ** 2)/ 2.0) * ( term1 * term2 - term3 * term4) / denom
    

    def _f12(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]

        return dt + (0.5 * (dt ** 2)) * (-np.sin(dTheta) * (2.0 * self.m2 * self.L1 * self.X[1] * np.cos(dTheta))) / (self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f13(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = np.cos(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - (np.sin(dTheta) ** 2) * (self.m2 * self.L1 * (self.X[1] ** 2)) - self.g * (-self.m2 * np.cos(self.X[0]))
        term3 = -2.0 * self.L1 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = -np.sin(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - self.g * (M * np.sin(self.X[0]) - self.m2 * np.sin(self.X[2]) * np.cos(dTheta))

        denom = term1 ** 2

        return (0.5 * (dt ** 2)) * ( term1 * term2 - term3 * term4) / denom

    
    def _f14(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]

        return (0.5 * (dt ** 2)) * (-2.0 * self.m2 * self.L2 * self.X[3] * np.sin(dTheta)) / (self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f21(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = -np.cos(dTheta) * ( self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - (np.sin(dTheta) ** 2) * (self.m2 * self.L1 * (self.X[3] ** 2)) - self.g * ( M * np.cos(self.X[0]) + self.m2 * np.sin(self.X[2]) * np.sin(dTheta))
        term3 = 2.0 * self.L1 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = -np.sin(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - self.g * (M * np.sin(self.X[0]) - self.m2 * np.sin(self.X[2] * np.cos(dTheta)))

        denom = term1 ** 2

        return dt * (term1 * term2 - term3 * term4) / denom
    

    def _f22(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]

        return 1.0 + dt * (-2.0 * self.m2 * self.L1 * self.X[0] * np.sin(dTheta)) / (self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f23(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = np.cos(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - (np.sin(dTheta) ** 2) * (self.m2 * self.L1 * (self.X[1] ** 2)) - self.m2 * self.g * np.cos(self.X[0])
        term3 = -2.0 * self.L1 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = -np.sin(dTheta) * (self.m2 * self.L1 * (self.X[1] ** 2) * np.cos(dTheta) + self.m2 * self.L2 * (self.X[3] ** 2)) - self.g * (M * np.sin(self.X[0]) - self.m2 * np.sin(self.X[2]) * np.cos(dTheta))

        denom = term1 ** 2

        return dt * (term1 * term2 - term3 * term4) / denom
    

    def _f24(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]

        return -dt * (2.0 * self.m2 * self.L2 * self.X[3] * np.sin(dTheta)) / (self.L1 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f31(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = np.cos(dTheta) * ( M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) - (np.sin(dTheta) ** 2) * (self.m2 * self.L2 * (self.X[3] ** 2)) + self.g * (M * np.cos(dTheta + self.X[0]))
        term3 = 2.0 * self.L2 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = np.sin(dTheta) * (M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) + M * self.g * (np.sin(self.X[0]) * np.cos(dTheta) - np.sin(self.X[2]))

        denom = term1 ** 2

        return ( 0.5 * dt ** 2 ) * (term1 * term2 - term3 * term4) / denom
    

    def _f32(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        return (dt ** 2) * (np.sin(dTheta) * M * self.L1 * self.X[1]) / (self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f33(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = -np.cos(dTheta) * (M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) + (np.sin(dTheta) ** 2) * (self.m2 * self.L2 * (self.X[3] ** 2)) + M * self.g * (np.sin(self.X[0]) * np.sin(dTheta) - np.cos(self.X[2]))
        term3 = -2.0 * self.L2 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = np.sin(dTheta) * (M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) + M * self.g * (np.sin(self.X[0]) * np.cos(dTheta) - np.sin(self.X[2]))

        denom = term1 ** 2

        return 1.0 + ( 0.5 * (dt ** 2) ) * (term1 * term2 - term3 * term4) / denom
    

    def _f34(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]

        return dt + 0.5 * (dt ** 2) * (2.0 * np.sin(dTheta) * self.m2 * self.L2 * self.X[3] * np.cos(dTheta)) / (self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f41(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = np.cos(dTheta) * ( M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) - (np.sin(dTheta) ** 2) * (self.m2 * self.L2 * (self.X[3] ** 2)) - self.g * M * np.cos(dTheta + self.X[0])
        term3 = 2.0 * self.L2 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = np.sin(dTheta) * (M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) + M * self.g * (np.sin(self.X[0]) * np.cos(dTheta) + np.sin(self.X[2]))

        denom = term1 ** 2

        return dt * (term1 * term2 - term3 * term4) / denom
    

    def _f42(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        return dt * (2.0 * M * self.L1 * np.sin(dTheta)) / (self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))
    

    def _f43(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]
        M = self.m1 + self.m2

        term1 = self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2))
        term2 = -np.cos(dTheta) * (M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) + (np.sin(dTheta) ** 2) * (self.m2 * self.L2 * (self.X[3] ** 2)) + M * self.g * (np.sin(self.X[0]) * np.sin(dTheta) + np.cos(self.X[2]))
        term3 = -2.0 * self.L2 * self.m2 * np.sin(dTheta) * np.cos(dTheta)
        term4 = np.sin(dTheta) * (M * self.L1 * (self.X[1] ** 2) + self.m2 * self.L2 * (self.X[3] ** 2) * np.cos(dTheta)) + M * self.g * (np.sin(self.X[0]) * np.cos(dTheta) + np.sin(self.X[2]))

        denom = term1 ** 2

        return dt * (term1 * term2 - term3 * term4) / denom
    

    def _f44(self, dt) -> float:
        dTheta = self.X[0] - self.X[2]

        return 1.0 + dt * (2.0 * self.m2 * self.L2 * self.X[3] * np.sin(dTheta) * np.cos(dTheta)) / (self.L2 * (self.m1 + self.m2 * (np.sin(dTheta) ** 2)))


    def _constructF(self, dt) -> np.ndarray:
        return np.array([[self._f11(dt), self._f12(dt), self._f13(dt), self._f14(dt)], [self._f21(dt), self._f22(dt), self._f23(dt), self._f24(dt)], [self._f31(dt), self._f32(dt), self._f33(dt), self._f34(dt)], [self._f41(dt), self._f42(dt), self._f43(dt), self._f44(dt)]])


    def _constructPredStateVec(self, dt) -> np.ndarray:

        theta1dotdot, theta2dotdot = dp.thetadotdot(self.m1, self.L1, self.m2, self.L2, self.X[0], self.X[1], self.X[2], self.X[3], self.g)

        theta1    = self.X[0] + dt * self.X[1] + 0.5 * (dt ** 2) * theta1dotdot
        theta1dot = self.X[1] + dt * theta1dotdot
        theta2    = self.X[2] + dt * self.X[3] + 0.5 * (dt ** 2) * theta2dotdot
        theta2dot = self.X[3] + dt * theta2dotdot


        return np.array([theta1, theta1dot, theta2, theta2dot])


    def _constructH(self, sensor) -> np.ndarray:
        if sensor == 1:
            H = np.array([[1.0, 0.0, 0.0, 0.0]])
            R = self.R[0][0]

        elif sensor == 2:
            H = np.array([[0.0, 0.0, 1.0, 0.0]])
            R = self.R[1][1]
        else:
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0 , 0.0]])
            R = self.R

        return H, R


    def predict(self, t) -> np.ndarray:
        dt = t - self.timeUpdated
        Xpred = self._constructPredStateVec(dt)

        Xpred[0] = self._wrapAngle(Xpred[0])
        Xpred[2] = self._wrapAngle(Xpred[2])

        F = self._constructF(dt)

        Ppred = F @ self.P @ np.transpose(F) + self.Q

        return Xpred, Ppred


    def update(self, Z, t, sensor) -> None:

        Xpred, Ppred = self.predict(t)

        H, R = self._constructH(sensor)
        Y = Z - H @ Xpred
        S = H @ Ppred @ np.transpose(H) + R
        K = Ppred @ np.transpose(H) @ np.linalg.inv(S)

        self.X = Xpred + K @ Y

        self.X[0] = self._wrapAngle(self.X[0])
        self.X[2] = self._wrapAngle(self.X[2])

        self.P = (np.identity(4) - K @ H) @ Ppred

        self.timeUpdated = t


    def newData(self, Z, t, sensor = 0) -> None:

        self.update(Z, t, sensor)

        self.timeHist.append(t)
        self.theta1Hist.append(self.X[0])
        self.theta2Hist.append(self.X[2])


    def _wrapAngle(self, angle) -> float:

        ratio = np.round(angle / (2.0 * np.pi))

        angle -= 2.0 * np.pi * ratio

        if angle >= np.pi:
            angle -= 2.0 * np.pi

        elif angle < -np.pi:
            angle += 2.0 * np.pi

        return angle