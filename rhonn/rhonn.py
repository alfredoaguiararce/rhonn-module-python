# -*- coding: utf-8 -*-
"""
Recurrent High Order Neural Networks (RHONN).
@author: Alfredo Aguiar Arce. <alfredoaguiararce@gmail.com> (www.alfredoagrar.com) 

Module to model neurons based on RHONN model.

Input variable Z     :   Measurements entering the neuron. Z = [ z1, z2, z3 ... zn]
Input variable W     :   Synaptic weight for its respective input Z. W = [w1, w2, w3 ... wn].Transpose
Output prediction    :   The result from the vectorial equation  (Z * W).
state EKF            :   State of the weights (bias) for each input Z.
"""

from abc import ABC, abstractmethod
import numpy as np
import math

class RHONN:
    def __init__(self, weights: int, zinputs: int):
        """
        Creates a neuron object based on RHONN.
        Inputs
        :param  : weights   : The array that contains initial values of the weights. W = [w1, w2, w3 ... wn]
        :param  : zinputs   : The array that contains signals or inputs for the neuron. Z = [ z1, z2, z3 ... zn]      
        """
        # Probe if W and Z has not the same number of elements in array.
        if np.size(weights) != np.size(zinputs):
            # The dimensions of weights and zinputs are diferents, raise an error. the size of W and Z need to be equals.
            raise ValueError("The dimensions of the vectors must be equal; size(weights) != size(zinputs)")
        
        # (int) number used for control of dimensions.
        self.size_input = np.size(zinputs)
        self.weights = np.array([weights]).T
        self.zinputs = np.array([zinputs])
        # init the ekf for the neuron
        self.ekf = EKF(self.weights, self.zinputs)
        # init the prediction as 0.
        self.prediction = 0

    def set_ekf(self, P_init : float =0.001, Q_init : float =0.001, R_init : float =0.001, FO: float=1):
        """
        Init the Extended Kalman Filter.
        Inputs
        :param  : P_init  : initial value for P matrix in the EKF.
        :param  : Q_init  : initial value for Q matrix in the EKF.
        :param  : R_init  : initial value for R in the EKF.
        :param  : FO      : Forgetting factor FO used to adjust the reaction to sudden changes in the forecast, if FO = 1 the EKFFO results must be equal to normal EKF.
        """
        self.ekf.init_filter(self.size_input, P_init, Q_init, R_init, FO)

    def update(self, new_zinputs, measure_signal):
        """
        Update the Extended Kalman Filter value in k-th
        Inputs
        :param  : new_zinputs     : The measurements of the inputs in the step k-th.
        :param  : measure_signal  : The value to predict. Used for calculate the error.
        """
        
        # Update the error by passing the measure value, and the previus value of prediction.
        self.ekf.set_error(measure_signal, self.predict())

        # check if are the same dimension from the (k - 1) iteration.
        # The size of the new z inputs dimensions, are different in this iteration.
        if np.size(new_zinputs) != np.size(self.zinputs):
            raise ValueError("The dimensions of the vector Z are diferent in this iteration.")
        self.zinputs = np.array([new_zinputs])
        self.weights = self.ekf.update_weights(self.zinputs)            
            
    
    def predict(self):
        """
        Calculate the value of the neuron in (k + 1)
        Return:
        :param : prediction : Numpy vectorial dot product of equation Z * W.
        """
        self.prediction = np.dot(self.zinputs, self.weights)
        return self.prediction

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



# The class ActivationFunction is an abstract base class that defines a method activate().
class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, c):
        pass

class SoftSigmoid(ActivationFunction):
    def activate(self, c, B=0.83):
        """
        Activation function soft sigmoidal.
        Inputs
        :param  : c    : Value of the signal in k-th
        :param  : B    : Beta used for the adjustment of the sigmoidal.

        Return:
        Result of the operation 1/(1 + e^(-B * C)) 
        """
        return 1 / (1 + math.exp(-B * c))
