## Double Pendulum Class

import numpy as np

class DoublePendulum:
    def __init__(self, inp_length1, inp_mass1, inp_length2, inp_mass2, inp_x1_0 = 0, inp_x1dot_0 = 0, inp_x2_0 = 0, inp_x2dot_0 = 0, inp_Mfric1 = 0, inp_Mfric2 = 0) -> None:

        # System Parameters
        self.length1 = inp_length1
        self.length2 = inp_length2
        self.mass1   = inp_mass1
        self.mass2   = inp_mass2
        self.Mfric1  = inp_Mfric1
        self.Mfric2  = inp_Mfric2

        # Time
        self.t = [0.0]

        # State Vector
        self.X = np.array([inp_x1_0, inp_x1dot_0, inp_x2_0, inp_x2dot_0])
        self.X_history = list(self.X)
        self.theta1_history = [self.X[0]]
        self.theta2_history = [self.X[2]]

    
    def _theta1Ode(self, x1, x1dot, x2, x2dot) -> float:
        # Constants
        dTheta = x1 - x2
        M = self.mass1 + self.mass2
        g = 9.81

        # Derivatives
        x1dotdot = ( 1.0 / (self.length1 * (self.mass1 + self.mass2 * (np.sin(dTheta) ** 2))) ) * ( -np.sin(dTheta) * ( self.mass2 * self.length1 * (x1dot ** 2) * np.cos(dTheta) + self.mass2 * self.length2 * (x2dot ** 2) ) - g * (M * np.sin(x1) - self.mass2 * np.sin(x2) * np.cos(dTheta) ) )

        # Return
        return x1dotdot

    
    def _theta2Ode(self, x1, x1dot, x2, x2dot) -> float:
        # Constants
        dTheta = x1 - x2
        M = self.mass1 + self.mass2
        g = 9.81

        # Derivatives
        x2dotdot = ( 1.0 / (self.length2 * (self.mass1 + self.mass2 * (np.sin(dTheta) ** 2))) ) * ( np.sin(dTheta) * ( M * self.length1 * (x1dot ** 2) + self.mass2 * self.length2 * (x2dot ** 2) * np.cos(dTheta)) + M * g * (np.sin(x1) * np.cos(dTheta) - np.sin(x2) ) )

        # Return
        return x2dotdot
    
    def _RungeKutta4thOrder(self, dt) -> None:
        
        dx1dot_1 = dt * self._theta1Ode(self.X[0], self.X[1], self.X[2], self.X[3])
        dx2dot_1 = dt * self._theta2Ode(self.X[0], self.X[1], self.X[2], self.X[3])
        dx1_1    = dt * self.X[1]
        dx2_1    = dt * self.X[3]

        dx1dot_2 = dt * self._theta1Ode(self.X[0] + dx1_1 / 2.0, self.X[1] + dx1dot_1 / 2.0, self.X[2] + dx2_1 / 2.0, self.X[3] + dx2dot_1 / 2.0)
        dx2dot_2 = dt * self._theta2Ode(self.X[0] + dx1_1 / 2.0, self.X[1] + dx1dot_1 / 2.0, self.X[2] + dx2_1 / 2.0, self.X[3] + dx2dot_1 / 2.0)
        dx1_2    = dt * (self.X[1] + dx1dot_1 / 2.0)
        dx2_2    = dt * (self.X[3] + dx2dot_1 / 2.0)

        dx1dot_3 = dt * self._theta1Ode(self.X[0] + dx1_2 / 2.0, self.X[1] + dx1dot_2 / 2.0, self.X[2] + dx2_2 / 2.0, self.X[3] + dx2dot_2 / 2.0)
        dx2dot_3 = dt * self._theta2Ode(self.X[0] + dx1_2 / 2.0, self.X[1] + dx1dot_2 / 2.0, self.X[2] + dx2_2 / 2.0, self.X[3] + dx2dot_2 / 2.0)
        dx1_3    = dt * (self.X[1] + dx1dot_2 / 2.0)
        dx2_3    = dt * (self.X[3] + dx2dot_2 / 2.0)

        dx1dot_4 = dt * self._theta1Ode(self.X[0] + dx1_3, self.X[1] + dx1dot_3, self.X[2] + dx2_3, self.X[3] + dx2dot_3)
        dx2dot_4 = dt * self._theta2Ode(self.X[0] + dx1_3, self.X[1] + dx1dot_3, self.X[2] + dx2_3, self.X[3] + dx2dot_3)
        dx1_4    = dt * (self.X[1] + dx1dot_3)
        dx2_4    = dt * (self.X[3] + dx2dot_3)

        self.X[0] = self.X[0] + (dx1_1 + 2.0 * dx1_2 + 2.0 * dx1_3 + dx1_4) / 6.0
        self.X[1] = self.X[1] + (dx1dot_1 + 2.0 * dx1dot_2 + 2.0 * dx1dot_3 + dx1dot_4) / 6.0
        self.X[2] = self.X[2] + (dx2_1 + 2.0 * dx2_2 + 2.0 * dx2_3 + dx2_4) / 6.0
        self.X[3] = self.X[3] + (dx2dot_1 + 2.0 * dx2dot_2 + 2.0 * dx2dot_3 + dx2dot_4) / 6.0

        self._checkAngleWrapping()

    def _checkAngleWrapping(self) -> None:

        ratio = np.floor(self.X[0] / (2.0 * np.pi))

        self.X[0] -= 2.0 * np.pi * ratio

        if self.X[0] >= np.pi:
            self.X[0] -= 2.0 * np.pi

        elif self.X[0] < -np.pi:
            self.X[0] += 2.0 * np.pi

        ratio = np.floor(self.X[2] / (2.0 * np.pi))

        self.X[2] -= 2.0 * np.pi * ratio

        if self.X[2] >= np.pi:
            self.X[2] -= 2.0 * np.pi

        elif self.X[2] < -np.pi:
            self.X[2] += 2.0 * np.pi
        

    def compute(self, duration, dt):
        # Time Vector
        T = np.arange(0, duration, dt)

        # Compute Double Pendulum
        for i in range(len(T) - 1):
            
            self._RungeKutta4thOrder(dt)

            self.X_history.append(self.X)
            self.theta1_history.append(self.X[0])
            self.theta2_history.append(self.X[2])
            
            self.t.append(T[i])