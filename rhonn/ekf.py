
import numpy as np
import math

class EKF: 
    """
    - NOTES - In this case for Extended Kalman Filter W, H, P, R, Q represent the values of k-th
    Wk, Pk represent the values of k + 1 : 

    - Li - is the total number of weights (w11,w12 ... wn)
    - m -  the total number of outputs in this case equals to 1 (X(k+1))
 
    - H - is a matrix  Li x m  
    - K - is a matrix  Li x m   
    - M - is a matrix  m  x m
    - P - is a matrix  Li x Li 
    - Q - is a matrix  Li x Li 
    - R - is a matrix  m  x m   
    - W - is a vector of Li elements Li x 1 (weights)

    - FO - is the forgettering factor if FO = 1 The model FKEFO transforms in a normal form of FKE.
    """

    def __init__(self, weights, H, FO=1):
        """
        Initialize the Extended Kalman Filter for the i-th neuron.
        Inputs
        :param  : weights   : Numpy array of actual vector W in the object rhonn.
        :param  : H         : Input vector used for the system linearizations.
        :param  : FO        : Forgetting factor, default is 1.
        """
        if np.size(weights) != np.size(H) or np.size(weights) <= 0:
            raise ValueError("Invalid dimensions for weights and H")

        self.Li = np.size(weights)
        self.m = 1
        self.H = H
        self.K = np.array(np.zeros((self.Li, self.m)))
        self.M = np.array(np.zeros((self.m, self.m)))
        self.P = np.array(np.zeros((self.Li, self.Li)))
        self.Q = np.array(np.zeros((self.Li, self.Li)))
        self.R = np.array(np.zeros((self.m, self.m)))
        self.W = weights
        self.error = 0
        self.fo = FO

    def init_filter(self, size_input: int, P_init, Q_init, R_init, FO):
        # Initialize EKF parameters
        # Declare an array of [0... n] elements where, each element are equal to init values.
        parray = np.array([P_init for x in range(0, size_input)])
        qarray = np.array([Q_init for x in range(0, size_input)])
        rarray = np.array([R_init])

        # P, Q, R must be diagonal matrix.
        P0 = np.diag(parray) 
        Q0 = np.diag(qarray) 
        R0 = np.diag(rarray)

        # init the paramethers in extended kalman filter (EKF) object.
        self.set_P(P0)
        self.set_Q(Q0)
        self.set_R(R0)
        self.set_FO(FO)

    def update_weights(self, H):
        """
        Update the values of bias W
        Inputs
        :param  : H : The vector H used for reference the linealitations of the system, in the k-th iteration. 

        Return:
        :param  : W : the ajust of the W vector as a new value of W. 

        """      
        # Declare the new value of H  
        self.H = H

        # M(k) = R * FO + H * P * H.T
        self.M = np.dot(self.R, self.fo) + (np.dot(self.H, np.dot(self.P, self.H.T)))

        num_rows, num_cols = self.M.shape
        if (num_rows == self.m) and (num_cols == self.m):
            # K(k) = (P * H.T) / M
            self.K = np.divide(np.dot(self.P, self.H.T), (self.M))
        else:
            raise ValueError('something happend M', num_rows, num_cols)


        num_rows, num_cols = self.K.shape
        if (num_rows == self.Li) and (num_cols == self.m):
            # W(k+1) = W(k) + K * error
            # wk is the actual value of W before the ejecution W(k)
            wk = self.W
            self.W = wk + (np.dot(self.K, self.error))
        else:
            raise ValueError('something happend K')


        # P(k+1) = P(K) * FO - (K * FO * H * P(k)) + Q
        # pk is the actual value of P before the ejecution P(k).
        pk = self.P
        self.P = (np.dot(pk, 1 / self.fo)) - (np.dot( np.dot(self.K, 1/ self.fo), np.dot(self.H, pk))) + self.Q

        # Return the estimation of W.
        return self.W

    def set_error(self, measure, estimation):
        """
        Track the error for the signal.
        Inputs
        :param  : measure     : It is the follow-up of the signal that you want to predict
        :param  : estimation  : Is the previous prediction of X(k + 1)
        """
        self.error = measure -  estimation

    # Def getters and setters for the variables of EKF
    def set_FO(self, FO):
        self.fo = FO
        
    def set_H(self, H):
        self.H = H

    def get_H(self):
        return self.H 
    
    def set_K(self, K):
        self.K = K

    def get_K(self):
        return self.K

    def set_M(self, M):
        self.M = M

    def get_M(self):
        return self.M

    def set_P(self, P):
        self.P = P

    def get_P(self):
        return self.P
    
    def set_Q(self, Q):
        self.Q = Q

    def get_Q(self):
        return self.Q

    def set_R(self, R):
        self.R = R

    def get_R(self):
        return self.R
    
    def set_W(self, W):
        self.W = W

    def get_W(self):
        return self.W

