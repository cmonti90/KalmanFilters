## Double Pendulum Class

import numpy as np


'''
The double pendulum is modeled with theta1 and theta2 as the angles of the first and second pendulum respectively.
theta1 is measured from the verticle axis with the positive direction being counter-clockwise and zero at the verticle axis pointing downwards.
theta2 is measured from theta1 with the positive direction being counter-clockwise and zero when the second pendulum is aligned with the first pendulum.

This implies that x is pointing downwards and y is pointing to the right.
'''


def thetadotdot(mass1, length1, mass2, length2, theta1, theta1dot, theta2, theta2dot, g = 9.81 ):
    
    # Common
    dTheta = theta1 - theta2
    M = mass1 + mass2
    denomFactor = mass1 + mass2 * (np.sin(dTheta) ** 2)


    # x1dotdot
    num = -np.sin(dTheta) * ( mass2 * length1 * (theta1dot ** 2) * np.cos(dTheta) + mass2 * length2 * (theta2dot ** 2) ) - g * ( M * np.sin(theta1) - mass2 * np.sin(theta2) * np.cos(dTheta) )
    denom = length1 * denomFactor

    theta1dotdot = num / denom

    # x2dotdot
    num = np.sin(dTheta) * ( M * length1 * (theta1dot ** 2) + mass2 * length2 * (theta2dot ** 2) * np.cos(dTheta)) + M * g * (np.sin(theta1) * np.cos(dTheta) - np.sin(theta2) )
    denom = length2 * denomFactor

    theta2dotdot = num / denom


    # Return
    return theta1dotdot, theta2dotdot


def wrapNegPiToPi(angle):
    
    if np.isscalar(angle):
        angle = angle % (2.0 * np.pi)
        if angle >= np.pi:
            angle -= 2.0 * np.pi
        elif angle < -np.pi:
            angle += 2.0 * np.pi
    else:
        # Handle the input as an array
        angle = np.mod(angle, 2.0 * np.pi)
        angle = np.where(angle >= np.pi, angle - 2.0 * np.pi, angle)
        angle = np.where(angle < -np.pi, angle + 2.0 * np.pi, angle)
    
    return angle


def wrapZeroToTwoPi(angle):
    return angle - 2.0 * np.pi * np.floor(angle / (2.0 * np.pi))


    

def findMinAngleDifference(angle1, angle2):

    dAng1 = angle1 - angle2
    dAng2 = dAng1 - 2 * np.pi * np.sign(dAng1)

    if np.isscalar(dAng1):
        return dAng1 if np.abs(dAng1) < np.abs(dAng2) else dAng2
    
    else:
        return np.where(np.abs(dAng1) < np.abs(dAng2), dAng1, dAng2)
        

def convertThetasToPos(theta1, theta2, length1, length2):
    '''
    x is pointing to the right
    y is pointing up
    '''

    x1 = length1 * np.cos(theta1)
    y1 = length1 * np.sin(theta1)

    x2 = x1 + length2 * np.cos(theta1 + theta2)
    y2 = y1 + length2 * np.sin(theta1 + theta2)

    x1_rot = y1
    y1_rot = -x1

    x2_rot = y2
    y2_rot = -x2

    return x1_rot, y1_rot, x2_rot, y2_rot



class DoublePendulum:
    def __init__(self, inp_length1, inp_mass1, inp_length2, inp_mass2, inp_x1_0 = 0, inp_x1dot_0 = 0, inp_x2_0 = 0, inp_x2dot_0 = 0, inp_g = 9.81):

        # System Parameters
        self.length1 = inp_length1
        self.length2 = inp_length2
        self.mass1   = inp_mass1
        self.mass2   = inp_mass2
        self.g       = inp_g

        # Time
        self.t = [0.0]

        # State Vector
        self.X = np.array([inp_x1_0, inp_x1dot_0, inp_x2_0, inp_x2dot_0])
        self.X_history = list(self.X)
        self.theta1_history = [self.X[0]]
        self.theta1dot_history = [self.X[1]]
        self.theta2_history = [self.X[2]]
        self.theta2dot_history = [self.X[3]]
    

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
            self.theta1dot_history.append(self.X[1])
            self.theta2_history.append(self.X[2])
            self.theta2dot_history.append(self.X[3])
            
            self.t.append(t)