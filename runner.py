import numpy as np
import matplotlib.pyplot as plt
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
    x1_0    = 10.0 * np.pi / 180.0
    x1dot_0 = 0.0 * np.pi / 180.0
    x2_0    = 0.0 * np.pi / 180.0
    x2dot_0 = 0.0 * np.pi / 180.0

    g = 9.81

    # Time
    duration = 20
    dt = 0.001

    # Instantiate object
    dubPen = dp.DoublePendulum(l1, m1, l2, m2, x1_0, x1dot_0, x2_0, x2dot_0, 0.0, 0.0, g)
    dubPen.compute(duration, dt)

    useEkf = 1
    useUkf = 1

    measRate = 1

    sensorNoise = ((1e-3)/3)**2
    useTruth = 0

    # Define the initial covariance
    sig_theta1dot_theta1dot = 0.0001
    sig_theta2dot_theta2dot = 0.0001
    sig_theta1_theta1 = 0.0001
    sig_theta2_theta2 = 0.0001

    sig_theta1dot_theta1 = 0
    sig_theta1dot_theta2 = 0
    sig_theta1dot_theta2dot = 0

    sig_theta2dot_theta2 = 0
    sig_theta2dot_theta1 = 0

    sig_theta1_theta2 = 0
    sig_theta1_theta2dot = 0

    P = np.array([[sig_theta1_theta1, sig_theta1dot_theta1, sig_theta1_theta2, sig_theta2dot_theta1], [sig_theta1dot_theta1, sig_theta1dot_theta1dot, sig_theta1dot_theta2, sig_theta1dot_theta2dot], [sig_theta1_theta2, sig_theta1dot_theta2, sig_theta2_theta2, sig_theta2dot_theta2], [sig_theta2dot_theta1, sig_theta1dot_theta2dot, sig_theta2dot_theta2, sig_theta2dot_theta2dot]])
    
    # Define process noise
    # Q = np.array([[0.0001, 0.00002, 0.00002, 0.0002], [0.00002, 0.0001, 0.00002, 0.00002], [0.00002, 0.00002, 0.0001, 0.00002], [0.00002, 0.00002, 0.00002, 0.0001]])
    Q = np.array([[0.0001, 0.0, 0.0, 0.0], [0.0, 0.0001, 0.0, 0.0], [0.0, 0.0, 0.0001, 0.0], [0.0, 0.0, 0.0, 0.0001]])

    # Define measurement noise
    R = np.array([[sensorNoise, 0.0], [0.0, sensorNoise]])

    # Define the initial state vector
    X = np.array([[x1_0], [x1dot_0], [x2_0], [x2dot_0]])

    # Instantiate the Kalman filters
    EKF = ekf.ExtendedKalmanFilter(m1, m2, l1, l2, P, X, R, Q, g)
    UKF = ukf.UnscentedKalmanFilter(m1, m2, l1, l2, P, X, R, Q, g, alpha = 1e-3, beta = 2, kappa = 0)

    ## Extended Kalman Filter

    timehist = [dubPen.t[0]]
    ekf_error_theta1 = [0.0]
    ekf_error_theta2 = [0.0]
    ukf_error_theta1 = [0.0]
    ukf_error_theta2 = [0.0]
    
    sensor = 1

    for i in range(1, len(dubPen.t)):

        # Time
        t = dubPen.t[i]
        timehist.append(t)


        # Define the measurement vector
        if useTruth:
            Z = np.array([[dubPen.theta1_history[i]], [dubPen.theta2_history[i]]])

        else:
            theta1_meas = dubPen.theta1_history[i] + np.random.normal(0.0, sensorNoise)
            theta2_meas = dubPen.theta2_history[i] + np.random.normal(0.0, sensorNoise)
            Z = np.array([[theta1_meas], [theta2_meas]])



        if i % measRate == 0:

            if sensor == 2:
                Z = np.array([Z[0]])
                sensor = 1
            elif sensor == 1:
                Z = np.array([Z[1]])
                sensor = 2


            if useEkf:
                
                # Update the Kalman filter
                EKF.newData(Z, t, sensor)

                ekf_error_theta1.append((EKF.theta1Hist[-1] - dubPen.theta1_history[i]) * 180.0 / np.pi )
                ekf_error_theta2.append((EKF.theta2Hist[-1] - dubPen.theta2_history[i]) * 180.0 / np.pi )


            if useUkf:

                # Update the Kalman filter
                UKF.newData(Z, t, sensor)

                ukf_error_theta1.append((UKF.theta1Hist[-1] - dubPen.theta1_history[i]) * 180.0 / np.pi )
                ukf_error_theta2.append((UKF.theta2Hist[-1] - dubPen.theta2_history[i]) * 180.0 / np.pi )


    plt.figure()
    plt.plot(dubPen.t, list(map(lambda x: x * 180 / np.pi, dubPen.theta1_history)), label="theta1")
    plt.plot(dubPen.t, list(map(lambda x: x * 180 / np.pi, dubPen.theta2_history)), label="theta2")

    plt.legend()
    plt.title("Double Pendulum Time Series: Truth")
    plt.xlabel("Time [sec]")
    plt.ylabel("Position [deg]")

    if useEkf:
        plt.figure()
        plt.plot(EKF.timeHist, ekf_error_theta1, label="theta1")
        plt.plot(EKF.timeHist, ekf_error_theta2, label="theta2")

        plt.legend()
        plt.title("Double Pendulum Time Series: EKF")
        plt.xlabel("Time [sec]")
        plt.ylabel("Error [deg]")

    if useUkf:
        plt.figure()
        plt.plot(UKF.timeHist, ukf_error_theta1, label="theta1")
        plt.plot(UKF.timeHist, ukf_error_theta2, label="theta2")

        plt.legend()
        plt.title("Double Pendulum Time Series: UKF")
        plt.xlabel("Time [sec]")
        plt.ylabel("Error [deg]")


        # plt.figure()
        # plt.plot(UKF.timeHist, list(map(lambda x: x * 180/np.pi, UKF.theta1Hist)), label="theta1")
        # plt.plot(UKF.timeHist, list(map(lambda x: x * 180/np.pi, UKF.theta2Hist)), label="theta2")

        # plt.legend()
        # plt.title("Double Pendulum Time Series: UKF")
        # plt.xlabel("Time [sec]")
        # plt.ylabel("Position [deg]")

    plt.show()