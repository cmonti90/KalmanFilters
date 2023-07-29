import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import math
import MackeyGlass as mg
import DoublePendulum as dp
import ExtendedKalmanFilter as ekf
import UnscentedKalmanFilter as ukf

if __name__ == "__main__":
    # Double pendulum parameters
    m1 = 3.0
    m2 = 1.0
    l1 = 2.0
    l2 = 1.0

    # Initial double pendulum states
    x1_0    = 0.0 * math.pi / 180.0
    x1dot_0 = 0.0
    x2_0    = 0.0
    x2dot_0 = 0.0

    # Time
    duration = 30.0
    dt = 0.001

    # Instantiate object
    dubPen = dp.DoublePendulum(l1, m1, l2, m2, x1_0, x1dot_0, x2_0, x2dot_0)
    dubPen.compute(duration, dt)

    useEkf = 0
    useUkf = 1


    # Define the initial covariance
    sig_theta1dot_theta1dot = 0.0001
    sig_theta2dot_theta2dot = 0.0001
    sig_theta1_theta1 = 0.0001
    sig_theta2_theta2 = 0.0001

    sig_theta1dot_theta1 = 0.0001
    sig_theta1dot_theta2 = 0.0001
    sig_theta1dot_theta2dot = 0.0001

    sig_theta2dot_theta2 = 0.0001
    sig_theta2dot_theta1 = 0.0001

    sig_theta1_theta2 = 0.0001
    sig_theta1_theta2dot = 0.0001

    P = np.array([[sig_theta1_theta1, sig_theta1dot_theta1, sig_theta1_theta2, sig_theta2dot_theta1], [sig_theta1dot_theta1, sig_theta1dot_theta1dot, sig_theta1dot_theta2, sig_theta1dot_theta2dot], [sig_theta1_theta2, sig_theta1dot_theta2, sig_theta2_theta2, sig_theta2dot_theta2], [sig_theta2dot_theta1, sig_theta1dot_theta2dot, sig_theta2dot_theta2, sig_theta2dot_theta2dot]])
    
    # Define process noise
    Q = np.array([[0.0001, 0.00002, 0.00002, 0.0002], [0.00002, 0.0001, 0.00002, 0.00002], [0.00002, 0.00002, 0.0001, 0.00002], [0.00002, 0.00002, 0.00002, 0.0001]])

    # Define measurement noise
    R = np.array([[1e-6, 0.0], [0.0, 1e-6]])

    # Define the initial state vector
    X = np.array([x1_0, x1dot_0, x2_0, x2dot_0])

    # Instantiate the Kalman filters
    EKF = ekf.ExtendedKalmanFilter(m1, m2, l1, l2, P, X, R, Q)
    UKF = ukf.UnscentedKalmanFilter(m1, m2, l1, l2, P, X, R, Q)

    ## Extended Kalman Filter

    offset = 1
    timehist = [dubPen.t[offset]]
    ekf_error_theta1 = [0.0]
    ekf_error_theta2 = [0.0]
    ukf_error_theta1 = [0.0]
    ukf_error_theta2 = [0.0]
    
    sensor = 1

    for i in range(len(dubPen.t) - offset):

        # Offset counter
        j = i + offset

        # Define the measurement vector

        theta1_meas = dubPen.theta1_history[j] + np.random.normal(0.0, 1e-4)
        theta2_meas = dubPen.theta2_history[j] + np.random.normal(0.0, 1e-4)
        Z = np.array([dubPen.theta1_history[j], dubPen.theta2_history[j]])
        # Z = np.array([theta1_meas, theta2_meas])

        # Time
        timehist.append(dubPen.t[j])
        t = dubPen.t[j]

        if useEkf:

            # sensor = 1

            if j % 25 == 0:

                if sensor == 2:
                    Z = Z[0]
                    sensor = 1
                elif sensor == 1:
                    Z = Z[1]
                    sensor = 2
                
                # Update the Kalman filter
                EKF.newData(Z, t, sensor)

                ekf_error_theta1.append((EKF.theta1Hist[-1] - dubPen.theta1_history[j]) * 180.0 / np.pi )
                ekf_error_theta2.append((EKF.theta2Hist[-1] - dubPen.theta2_history[j]) * 180.0 / np.pi )

        if useUkf:
            # Update the Kalman filter
            # try:
                UKF.newData(Z, t)

                ukf_error_theta1.append((UKF.theta1Hist[-1] - dubPen.theta1_history[j]) * 180.0 / np.pi )
                ukf_error_theta2.append((UKF.theta2Hist[-1] - dubPen.theta2_history[j]) * 180.0 / np.pi )

            # finally:
            #         print(f"Final EKF error theta1: {ekf_error_theta1[-1]}")
            #         print(f"Final EKF error theta2: {ekf_error_theta2[-1]}")
            #         print(f"Final UKF error theta1: {ukf_error_theta1[-1]}")
            #         print(f"Final UKF error theta2: {ukf_error_theta2[-1]}")
            #         break



    ## Unscented Kalman Filter



    ### Plot the results
    # Truth data
    # plt.plot(dubPen.t, dubPen.theta1_history, label="theta1_true")
    # plt.plot(dubPen.t, dubPen.theta2_history, label="theta2_true")

    # # EKF data
    # plt.plot(EKF.timeHist, EKF.theta1Hist, label="theta1_EKF")
    # plt.plot(EKF.timeHist, EKF.theta2Hist, label="theta2_EKF")

    # plt.legend()
    # plt.title("Double Pendulum Time Series")
    # plt.xlabel("Time")
    # plt.ylabel("Angle (deg)")
    # plt.show()

    if useEkf:
        plt.plot(EKF.timeHist, ekf_error_theta1, label="theta1")
        plt.plot(EKF.timeHist, ekf_error_theta2, label="theta2")

        plt.legend()
        plt.title("Double Pendulum Time Series: EKF")
        plt.xlabel("Time")
        plt.ylabel("Error (deg)")
        plt.show()

    if useUkf:
        plt.plot(UKF.timeHist, ukf_error_theta1, label="theta1")
        plt.plot(UKF.timeHist, ukf_error_theta2, label="theta2")

        plt.legend()
        plt.title("Double Pendulum Time Series: UKF")
        plt.xlabel("Time")
        plt.ylabel("Error (deg)")
        plt.show()