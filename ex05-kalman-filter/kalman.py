import numpy as np
import matplotlib.pyplot as plt

class Kalman:
    def __init__(self,A,B,C) -> None:
        self.A = A # process model
        self.B = B # control model
        self.C = C # measurement model

        self.Sigma = np.zeros((A.shape[0],A.shape[0])) # state covariance
        self.R = np.zeros((A.shape[0],A.shape[0]))  # process noise covariance
        self.Q = np.zeros((C.shape[0],C.shape[0]))  # measurement noise covariance
        self.x = np.zeros((A.shape[0],1)) # state vector
    
    def predict(self,u): # u is the control input
        self.x = self.A @ self.x + self.B @ u
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.R
        return self.x

    def update(self,z): # z is the measurement
        K = self.Sigma @ self.C.T @ np.linalg.inv(self.C @ self.Sigma @ self.C.T + self.Q)
        self.x = self.x + K @ (z - self.C @ self.x)
        self.Sigma = (np.eye(self.A.shape[0]) - K @ self.C) @ self.Sigma
        return self.x


if __name__ == "__main__":
    delta_t = 1
    A = np.array([[1, delta_t],[0,1]])
    B = np.array([[0.5*delta_t**2],[delta_t]])
    C = np.array([[1,0]])
    kalman = Kalman(A,B,C)

    sigma2_y = 0.25
    sigma2_v = 0.05
    simga2_z = 0.5
    kalman.Sigma = np.array([[0.1,0],[0,0.1]])
    kalman.R = np.array([[sigma2_y,0],[0,sigma2_v]])
    kalman.Q = np.array([[simga2_z]])
    kalman.x = np.array([[95.5],[0]])

    a_y = np.array([[-1]])
    z_input = [96.4,95.9,94.4,87.7,85.3]
    x_correct = []
    for z in z_input:
        kalman.predict(a_y)
        kalman.update(z)
        x_correct.append(kalman.x[0])
    plt.plot(z_input,label='measurement')
    plt.plot(x_correct,label='corrected')
    plt.show()