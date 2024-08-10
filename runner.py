import numpy as np
import matplotlib.pyplot as plt
import DoublePendulum as dp
import ExtendedKalmanFilter as ekf
import UnscentedKalmanFilter as ukf

showPlots = 1

montecarl0 = 0

np.random.seed(0)

# Environment
g = 9.81

# Double pendulum nominal parameters
m1 = 3.0
m2 = 1.0
l1 = 2.0
l2 = 1.0

# Monte carlo parameters
m1_sigma = 0.1
m2_sigma = 0.03
l1_sigma = 0.05
l2_sigma = 0.03

# Initial double pendulum states
theta1_0    = 10.0 * np.pi / 180.0
theta1dot_0 = 0.0 * np.pi / 180.0
theta2_0    = -30.0 * np.pi / 180.0
theta2dot_0 = 0.0 * np.pi / 180.0

# Time
duration = 20
dt_model = 0.0001

dt_ratio = 10


# Filters
useEkf = 1
useUkf = 1

measRate = 100

sensorNoise1Sigma = 0.5 * np.pi/180
useTruth = 0

sensor = 0


# Double pendulum true parameters
m1_true = m1 + montecarl0 * np.random.normal(0.0, m1_sigma)
m2_true = m2 + montecarl0 * np.random.normal(0.0, m2_sigma)
l1_true = l1 + montecarl0 * np.random.normal(0.0, l1_sigma)
l2_true = l2 + montecarl0 * np.random.normal(0.0, l2_sigma)

# Instantiate object
dubPen = dp.DoublePendulum(l1_true, m1_true, l2_true, m2_true, theta1_0, theta1dot_0, theta2_0, theta2dot_0, g)
dubPen.compute(duration, dt_model)

# Define the initial covariance
sig_theta1dot_theta1dot = (0.001 * np.pi / 180.0)**2
sig_theta2dot_theta2dot = (0.001 * np.pi / 180.0)**2
sig_theta1_theta1 = (0.01 * np.pi / 180.0)**2
sig_theta2_theta2 = (0.01 * np.pi / 180.0)**2

sig_theta1dot_theta1 = 0
sig_theta1dot_theta2 = 0
sig_theta1dot_theta2dot = 0

sig_theta2dot_theta2 = 0
sig_theta2dot_theta1 = 0

sig_theta1_theta2 = 0
sig_theta1_theta2dot = 0

P0 = np.array([[sig_theta1_theta1, sig_theta1dot_theta1, sig_theta1_theta2, sig_theta2dot_theta1],
               [sig_theta1dot_theta1, sig_theta1dot_theta1dot, sig_theta1dot_theta2, sig_theta1dot_theta2dot],
               [sig_theta1_theta2, sig_theta1dot_theta2, sig_theta2_theta2, sig_theta2dot_theta2],
               [sig_theta2dot_theta1, sig_theta1dot_theta2dot, sig_theta2dot_theta2, sig_theta2dot_theta2dot]])


ekf_var_theta1dotdot = (22 * np.pi / 180)**2
ekf_var_theta2dotdot = (22 * np.pi / 180)**2
ekf_cov_theta1dotdot_theta2dotdot = 0

# ukf_var_theta1dotdot = (45 * np.pi / 180)**2
# ukf_var_theta2dotdot = (500 * np.pi / 180)**2
ukf_var_theta1dotdot = (22 * np.pi / 180)**2
ukf_var_theta2dotdot = (22 * np.pi / 180)**2
ukf_cov_theta1dotdot_theta2dotdot = 0

def chooseQ( inp_var_theta1dotdot, inp_var_theta2dotdot, inp_cov_theta1dotdot_theta2dotdot ):
    def computeQ(dt):

        dt2 = dt**2
        dt3 = dt * dt2
        dt4 = dt * dt3
        
        Q = np.zeros((4,4))

        Q[0,0] = 0.25 * dt4 * inp_var_theta1dotdot
        Q[1,1] = dt2 * inp_var_theta1dotdot
        Q[2,2] = 0.25 * dt4 * inp_var_theta2dotdot
        Q[3,3] = dt2 * inp_var_theta2dotdot

        Q[0,1] = 0.5 * dt3 * inp_var_theta1dotdot
        Q[0,2] = 0.25 * dt4 * inp_cov_theta1dotdot_theta2dotdot
        Q[0,3] = 0.5 * dt3 * inp_cov_theta1dotdot_theta2dotdot

        Q[1,0] = Q[0,1]
        Q[1,2] = 0.5 * dt3 * inp_cov_theta1dotdot_theta2dotdot
        Q[1,3] = dt2 * inp_cov_theta1dotdot_theta2dotdot
        
        Q[2,0] = Q[0,2]
        Q[2,1] = Q[1,2]
        Q[2,3] = 0.5 * dt3 * inp_var_theta2dotdot

        Q[3,0] = Q[0,3]
        Q[3,1] = Q[1,3]
        Q[3,2] = Q[2,3]

        return Q
    
    return computeQ

Qekf = chooseQ( ekf_var_theta1dotdot, ekf_var_theta2dotdot, ekf_cov_theta1dotdot_theta2dotdot )
Qukf = chooseQ( ukf_var_theta1dotdot, ukf_var_theta2dotdot, ukf_cov_theta1dotdot_theta2dotdot )


R = np.array([[sensorNoise1Sigma**2, 0.0],
              [0.0, sensorNoise1Sigma**2]])

# Define the initial state vector
X0 = np.array([[theta1_0],
               [theta1dot_0],
               [theta2_0],
               [theta2dot_0]])

# Instantiate the Kalman filters
EKF = ekf.ExtendedKalmanFilter(m1, m2, l1, l2, P0, X0, R, Qekf, g)
UKF = ukf.UnscentedKalmanFilter(m1, m2, l1, l2, P0, X0, R, Qukf, g, alpha = 1e-3, beta = 2, kappa = 0)


timehist = [dubPen.t[0]]

ekf_theta1    = [theta1_0 * 180.0 / np.pi]
ekf_theta1dot = [theta1dot_0 * 180.0 / np.pi]
ekf_theta2    = [theta2_0 * 180.0 / np.pi]
ekf_theta2dot = [theta2dot_0 * 180.0 / np.pi]
ekf_error_theta1    = [0.0]
ekf_error_theta1dot = [0.0]
ekf_error_theta2    = [0.0]
ekf_error_theta2dot = [0.0]
ekf_P = [P0]
ekf_Ppred = [P0]

ukf_theta1    = [theta1_0 * 180.0 / np.pi]
ukf_theta1dot = [theta1dot_0 * 180.0 / np.pi]
ukf_theta2    = [theta2_0 * 180.0 / np.pi]
ukf_theta2dot = [theta2dot_0 * 180.0 / np.pi]
ukf_error_theta1    = [0.0]
ukf_error_theta1dot = [0.0]
ukf_error_theta2    = [0.0]
ukf_error_theta2dot = [0.0]
ukf_P = [P0]
ukf_Ppred = [P0]

t_meas = [0.0]

t_meas_theta1 = []
meas_theta1 = []
t_meas_theta2 = []
meas_theta2 = []

j = 0

for i in range(dt_ratio, len(dubPen.t), dt_ratio):

    j = j + 1

    # Time
    t = dubPen.t[i]
    timehist.append(t)


    # Define the measurement vector
    if useTruth:
        Z = np.array([[dubPen.theta1_history[i]], [dubPen.theta2_history[i]]])

    else:
        theta1_meas = dubPen.theta1_history[i] + np.random.normal(0.0, sensorNoise1Sigma)
        theta2_meas = dubPen.theta2_history[i] + np.random.normal(0.0, sensorNoise1Sigma)
        Z = np.array([[theta1_meas], [theta2_meas]])


    if j % measRate == 0:

        t_meas.append(t)

        if sensor == 2:
            Z = np.array([Z[0]])
            sensor = 1

            t_meas_theta1.append(t)
            meas_theta1.append(Z[0,0] * 180.0 / np.pi)

        elif sensor == 1:
            Z = np.array([Z[1]])
            sensor = 2

            t_meas_theta2.append(t)
            meas_theta2.append(Z[0,0] * 180.0 / np.pi)

        else:

            t_meas_theta1.append(t)
            meas_theta1.append(Z[0,0] * 180.0 / np.pi)

            t_meas_theta2.append(t)
            meas_theta2.append(Z[1,0] * 180.0 / np.pi)



        if useEkf:
            
            # Update the Kalman filter
            stateVec_ekf = EKF.update(t, Z, sensor)

            ekf_theta1   .append(stateVec_ekf[0,0] * 180.0 / np.pi)
            ekf_theta1dot.append(stateVec_ekf[1,0] * 180.0 / np.pi)
            ekf_theta2   .append(stateVec_ekf[2,0] * 180.0 / np.pi)
            ekf_theta2dot.append(stateVec_ekf[3,0] * 180.0 / np.pi)

            ekf_error_theta1   .append((stateVec_ekf[0,0] - dubPen.theta1_history   [i]) * 180.0 / np.pi )
            ekf_error_theta1dot.append((stateVec_ekf[1,0] - dubPen.theta1dot_history[i]) * 180.0 / np.pi )
            ekf_error_theta2   .append((stateVec_ekf[2,0] - dubPen.theta2_history   [i]) * 180.0 / np.pi )
            ekf_error_theta2dot.append((stateVec_ekf[3,0] - dubPen.theta2dot_history[i]) * 180.0 / np.pi )

            ekf_P.append(EKF.P)
            ekf_Ppred.append(EKF.P)


        if useUkf:

            # Update the Kalman filter
            stateVec_ukf = UKF.update(t, Z, sensor)

            ukf_theta1   .append(stateVec_ukf[0,0] * 180.0 / np.pi)
            ukf_theta1dot.append(stateVec_ukf[1,0] * 180.0 / np.pi)
            ukf_theta2   .append(stateVec_ukf[2,0] * 180.0 / np.pi)
            ukf_theta2dot.append(stateVec_ukf[3,0] * 180.0 / np.pi)

            ukf_error_theta1   .append((stateVec_ukf[0,0] - dubPen.theta1_history   [i]) * 180.0 / np.pi )
            ukf_error_theta1dot.append((stateVec_ukf[1,0] - dubPen.theta1dot_history[i]) * 180.0 / np.pi )
            ukf_error_theta2   .append((stateVec_ukf[2,0] - dubPen.theta2_history   [i]) * 180.0 / np.pi )
            ukf_error_theta2dot.append((stateVec_ukf[3,0] - dubPen.theta2dot_history[i]) * 180.0 / np.pi )

            ukf_P.append(UKF.P)
            ukf_Ppred.append(UKF.P)


    else:

        if useEkf:
            
            # Update the Kalman filter
            stateVec_ekf, Ppred_ekf = EKF.predict(t)

            ekf_theta1   .append(stateVec_ekf[0,0] * 180.0 / np.pi)
            ekf_theta1dot.append(stateVec_ekf[1,0] * 180.0 / np.pi)
            ekf_theta2   .append(stateVec_ekf[2,0] * 180.0 / np.pi)
            ekf_theta2dot.append(stateVec_ekf[3,0] * 180.0 / np.pi)

            ekf_error_theta1   .append((stateVec_ekf[0,0] - dubPen.theta1_history   [i]) * 180.0 / np.pi )
            ekf_error_theta1dot.append((stateVec_ekf[1,0] - dubPen.theta1dot_history[i]) * 180.0 / np.pi )
            ekf_error_theta2   .append((stateVec_ekf[2,0] - dubPen.theta2_history   [i]) * 180.0 / np.pi )
            ekf_error_theta2dot.append((stateVec_ekf[3,0] - dubPen.theta2dot_history[i]) * 180.0 / np.pi )

            ekf_Ppred.append(Ppred_ekf)


        if useUkf:

            # Update the Kalman filter
            stateVec_ukf, Ppred_ukf, _ = UKF.predict(t)

            ukf_theta1   .append(stateVec_ukf[0,0] * 180.0 / np.pi)
            ukf_theta1dot.append(stateVec_ukf[1,0] * 180.0 / np.pi)
            ukf_theta2   .append(stateVec_ukf[2,0] * 180.0 / np.pi)
            ukf_theta2dot.append(stateVec_ukf[3,0] * 180.0 / np.pi)

            ukf_error_theta1   .append((stateVec_ukf[0,0] - dubPen.theta1_history   [i]) * 180.0 / np.pi )
            ukf_error_theta1dot.append((stateVec_ukf[1,0] - dubPen.theta1dot_history[i]) * 180.0 / np.pi )
            ukf_error_theta2   .append((stateVec_ukf[2,0] - dubPen.theta2_history   [i]) * 180.0 / np.pi )
            ukf_error_theta2dot.append((stateVec_ukf[3,0] - dubPen.theta2dot_history[i]) * 180.0 / np.pi )

            ukf_Ppred.append(Ppred_ukf)


# Calculate root-mean-square of error terms
ekf_error_rms_theta1 = np.sqrt(np.mean(np.square(ekf_error_theta1)))
ekf_error_rms_theta1dot = np.sqrt(np.mean(np.square(ekf_error_theta1dot)))
ekf_error_rms_theta2 = np.sqrt(np.mean(np.square(ekf_error_theta2)))
ekf_error_rms_theta2dot = np.sqrt(np.mean(np.square(ekf_error_theta2dot)))

ukf_error_rms_theta1 = np.sqrt(np.mean(np.square(ukf_error_theta1)))
ukf_error_rms_theta1dot = np.sqrt(np.mean(np.square(ukf_error_theta1dot)))
ukf_error_rms_theta2 = np.sqrt(np.mean(np.square(ukf_error_theta2)))
ukf_error_rms_theta2dot = np.sqrt(np.mean(np.square(ukf_error_theta2dot)))


print("EKF RMS Error: theta1 = {:.4f} deg, theta1dot = {:.4f} deg/sec, theta2 = {:.4f} deg, theta2dot = {:.4f} deg/sec".format(ekf_error_rms_theta1, ekf_error_rms_theta1dot, ekf_error_rms_theta2, ekf_error_rms_theta2dot))
print("UKF RMS Error: theta1 = {:.4f} deg, theta1dot = {:.4f} deg/sec, theta2 = {:.4f} deg, theta2dot = {:.4f} deg/sec".format(ukf_error_rms_theta1, ukf_error_rms_theta1dot, ukf_error_rms_theta2, ukf_error_rms_theta2dot))



# thread = QThread()
# thread.start()
if showPlots:
    ## True Position
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(dubPen.t, list(map(lambda x: x * 180 / np.pi, dubPen.theta1_history)), label="Truth", linestyle='solid')
    axs[0].plot(timehist, ekf_theta1, label="EKF", linestyle='dashed')
    axs[0].plot(timehist, ukf_theta1, label="UKF", linestyle='dashed')
    axs[0].plot(t_meas_theta1, meas_theta1, '+', label="meas")
    
    axs[0].grid()

    axs[0].legend()
    axs[0].set_title(r"$\theta_{1}$")
    axs[0].set_xlabel("Time [sec]")
    axs[0].set_ylabel("Position [deg]")


    axs[1].plot(dubPen.t, list(map(lambda x: x * 180 / np.pi, dubPen.theta2_history)), label="Truth", linestyle='solid')
    axs[1].plot(timehist, ekf_theta2, label="EKF", linestyle='dashed')
    axs[1].plot(timehist, ukf_theta2, label="UKF", linestyle='dashed')
    axs[1].plot(t_meas_theta2, meas_theta2, '+', label="meas")
    
    axs[1].grid()


    axs[1].legend()
    axs[1].set_title(r"$\theta_{2}$")
    axs[1].set_xlabel("Time [sec]")
    axs[1].set_ylabel("Position [deg]")

    fig.tight_layout()

    # True Velocity
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(dubPen.t, list(map(lambda x: x * 180 / np.pi, dubPen.theta1dot_history)), linestyle='-', label="Truth")
    axs[0].plot(timehist, ekf_theta1dot, linestyle='--', label="EKF")
    axs[0].plot(timehist, ukf_theta1dot, linestyle='--', label="UKF")
    
    axs[0].grid()

    axs[0].legend()
    axs[0].set_title(r"$\dot{\theta}_{1}$")
    axs[0].set_xlabel("Time [sec]")
    axs[0].set_ylabel("Velocity [deg/sec]")


    axs[1].plot(dubPen.t, list(map(lambda x: x * 180 / np.pi, dubPen.theta2dot_history)), linestyle='-', label="Truth")
    axs[1].plot(timehist, ekf_theta2dot, linestyle='--', label="EKF")
    axs[1].plot(timehist, ukf_theta2dot, linestyle='--', label="UKF")
    
    axs[1].grid()


    axs[1].legend()
    axs[1].set_title(r"$\dot{\theta}_{2}$")
    axs[1].set_xlabel("Time [sec]")
    axs[1].set_ylabel("Velocity [deg/sec]")

    fig.tight_layout()

    ## Error
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(timehist, ekf_error_theta1, label="EKF")
    axs[0].plot(timehist, ukf_error_theta1, label="UKF")
    
    axs[0].grid()

    axs[0].legend()
    axs[0].set_title(r"$\theta_{1}$ Error")
    axs[0].set_xlabel("Time [sec]")
    axs[0].set_ylabel("Position Error [deg]")


    axs[1].plot(timehist, ekf_error_theta2, label="EKF")
    axs[1].plot(timehist, ukf_error_theta2, label="UKF")
    
    axs[1].grid()

    axs[1].legend()
    axs[1].set_title(r"$\theta_{2}$ Error")
    axs[1].set_xlabel("Time [sec]")
    axs[1].set_ylabel("Position Error [deg]")


    fig, axs = plt.subplots(2, 1)

    axs[0].plot(timehist, ekf_error_theta1dot, label="EKF")
    axs[0].plot(timehist, ukf_error_theta1dot, label="UKF")
    
    axs[0].grid()

    axs[0].legend()
    axs[0].set_title(r"$\dot{\theta}_{1}$ Error")
    axs[0].set_xlabel("Time [sec]")
    axs[0].set_ylabel("Velocity Error [deg/sec]")

    axs[1].plot(timehist, ekf_error_theta2dot, label="EKF")
    axs[1].plot(timehist, ukf_error_theta2dot, label="UKF")
    
    axs[1].grid()

    axs[1].legend()
    axs[1].set_title(r"$\dot{\theta}_{2}$ Error")
    axs[1].set_xlabel("Time [sec]")
    axs[1].set_ylabel("Velocity Error [deg/sec]")

    fig.tight_layout()


    # Covariance
    fig, axs = plt.subplots(4, 4)

    for i in range(4):
        for j in range(4):
            
            axs[i,j].plot(timehist, list(map(lambda x: x[i,j], ekf_Ppred)), ls = "--", color = "orange", label="EKF")
            axs[i,j].plot(timehist, list(map(lambda x: x[i,j], ukf_Ppred)), ls = "--", color = "g", label="UKF")

            axs[i,j].plot(t_meas, list(map(lambda x: x[i,j], ekf_P)), ls = "", marker = "+", mec = "orange")
            axs[i,j].plot(t_meas, list(map(lambda x: x[i,j], ukf_P)), ls = "", marker = "+", color = "g")

            axs[i,j].grid()

            axs[i,j].set_title("P[{},{}]".format(i,j))
            axs[i,j].set_xlabel("Time [sec]")
            axs[i,j].set_ylabel("Value")

            axs[i,j].sharex(axs[0,0])

        

    axs[0, 3].legend( loc = "upper right", bbox_to_anchor = (1.5, 1.0) )

    # fig.canvas.manager.window.showMaximized()

    fig.tight_layout()



    plt.show()
