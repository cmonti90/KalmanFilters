## Double Pendulum Class

import numpy as np


def thetadotdot(mass1, length1, mass2, length2, x1, x1dot, x2, x2dot, g = 9.81 ) -> float:
    
    # Common
    dTheta = x1 - x2
    M = mass1 + mass2
    denomFactor = mass1 + mass2 * (np.sin(dTheta) ** 2)


    # x1dotdot
    num = -np.sin(dTheta) * ( mass2 * length1 * (x1dot ** 2) * np.cos(dTheta) + mass2 * length2 * (x2dot ** 2) ) - g * ( M * np.sin(x1) - mass2 * np.sin(x2) * np.cos(dTheta) )
    denom = length1 * denomFactor

    x1dotdot = num / denom

    # x2dotdot
    num = np.sin(dTheta) * ( M * length1 * (x1dot ** 2) + mass2 * length2 * (x2dot ** 2) * np.cos(dTheta)) + M * g * (np.sin(x1) * np.cos(dTheta) - np.sin(x2) )
    denom = length2 * denomFactor

    x2dotdot = num / denom


    # Return
    return x1dotdot, x2dotdot


def wrapNegPiToPi(angle: np.float64) -> np.float64:

    ratio = np.trunc(angle / (2.0 * np.pi))

    angle -= 2.0 * np.pi * ratio

    if angle >= np.pi:
        angle -= 2.0 * np.pi

    elif angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def wrapZeroToTwoPi(angle: np.float64) -> np.float64:

    ratio = np.trunc(angle / (2.0 * np.pi))

    angle -= 2.0 * np.pi * ratio

    if angle < 0:
        angle += 2.0 * np.pi

    return angle


class DoublePendulum:
    def __init__(self, inp_length1, inp_mass1, inp_length2, inp_mass2, inp_x1_0 = 0, inp_x1dot_0 = 0, inp_x2_0 = 0, inp_x2dot_0 = 0, inp_Mfric1 = 0, inp_Mfric2 = 0, inp_g = 9.81) -> None:

        # System Parameters
        self.length1 = inp_length1
        self.length2 = inp_length2
        self.mass1   = inp_mass1
        self.mass2   = inp_mass2
        self.Mfric1  = inp_Mfric1
        self.Mfric2  = inp_Mfric2
        self.g       = inp_g

        # Time
        self.t = [0.0]

        # State Vector
        self.X = np.array([inp_x1_0, inp_x1dot_0, inp_x2_0, inp_x2dot_0])
        self.X_history = list(self.X)
        self.theta1_history = [self.X[0]]
        self.theta2_history = [self.X[2]]
    

    def _RungeKutta4thOrder(self, dt) -> None:
        
        x1dotdot, x2dotdot = thetadotdot(self.mass1, self.length1, self.mass2, self.length2, self.X[0], self.X[1], self.X[2], self.X[3], self.g)
        k1_x1dot = dt * x1dotdot
        k1_x2dot = dt * x2dotdot
        k1_x1    = dt * self.X[1]
        k1_x2    = dt * self.X[3]

        x1dotdot, x2dotdot = thetadotdot(self.mass1, self.length1, self.mass2, self.length2, self.X[0] + k1_x1 / 2.0, self.X[1] + k1_x1dot / 2.0, self.X[2] + k1_x2 / 2.0, self.X[3] + k1_x2dot / 2.0, self.g)
        k2_x1dot = dt * x1dotdot
        k2_x2dot = dt * x2dotdot
        k2_x1    = dt * (self.X[1] + k1_x1dot / 2.0)
        k2_x2    = dt * (self.X[3] + k1_x2dot / 2.0)

        x1dotdot, x2dotdot = thetadotdot(self.mass1, self.length1, self.mass2, self.length2, self.X[0] + k2_x1 / 2.0, self.X[1] + k2_x1dot / 2.0, self.X[2] + k2_x2 / 2.0, self.X[3] + k2_x2dot / 2.0, self.g)
        k3_x1dot = dt * x1dotdot
        k3_x2dot = dt * x2dotdot
        k3_x1    = dt * (self.X[1] + k2_x1dot / 2.0)
        k3_x2    = dt * (self.X[3] + k2_x2dot / 2.0)

        x1dotdot, x2dotdot = thetadotdot(self.mass1, self.length1, self.mass2, self.length2, self.X[0] + k3_x1, self.X[1] + k3_x1dot, self.X[2] + k3_x2, self.X[3] + k3_x2dot, self.g)
        k4_x1dot = dt * x1dotdot
        k4_x2dot = dt * x2dotdot
        k4_x1    = dt * (self.X[1] + k3_x1dot)
        k4_x2    = dt * (self.X[3] + k3_x2dot)


        self.X[0] += ( k1_x1    + 2.0 * ( k2_x1    + k3_x1    ) + k4_x1    ) / 6.0
        self.X[1] += ( k1_x1dot + 2.0 * ( k2_x1dot + k3_x1dot ) + k4_x1dot ) / 6.0
        self.X[2] += ( k1_x2    + 2.0 * ( k2_x2    + k3_x2    ) + k4_x2    ) / 6.0
        self.X[3] += ( k1_x2dot + 2.0 * ( k2_x2dot + k3_x2dot ) + k4_x2dot ) / 6.0


        self.X[0] = wrapNegPiToPi(self.X[0])
        self.X[2] = wrapNegPiToPi(self.X[2])
        

    def compute(self, duration, dt):

        # Compute Double Pendulum
        for t in np.arange(dt, duration, dt):
            
            self._RungeKutta4thOrder(dt)

            self.X_history.append(self.X)
            self.theta1_history.append(self.X[0])
            self.theta2_history.append(self.X[2])
            
            self.t.append(t)