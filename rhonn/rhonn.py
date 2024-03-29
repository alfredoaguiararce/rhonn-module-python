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

import numpy as np
import ekf

class rhonn:
    def __init__(self
                 , weights: list
                 , zinputs: list
                 ):
        """
        Creates a neuron object based on RHONN.
        Inputs
        :param  : weights   : The array that contains initial values of the weights. W = [w1, w2, w3 ... wn]
        :param  : zinputs   : The array that contains signals or inputs for the neuron. Z = [ z1, z2, z3 ... zn]      
        """
        if type(weights) is list:
            weights: np.ndarray = np.array(weights).flatten()
        
        if type(zinputs) is list:
            zinputs: np.ndarray = np.array(zinputs).flatten()

        # Probe if W and Z has not the same number of elements in array.
        if np.size(weights) != np.size(zinputs):
            # The dimensions of weights and zinputs are diferents, raise an error. the size of W and Z need to be equals.
            raise ValueError("The dimensions of the vectors must be equal; size(weights) != size(zinputs)")
        
        # This code snippet is checking if the `weights` and `zinputs` variables are numpy arrays. If
        # either of them is not a numpy array, it raises a `ValueError` with the message "The input
        # must be numpy arrays." This is done to ensure that the inputs to the `rhonn` class are numpy
        # arrays, as the code relies on numpy array operations.
        if type(weights) != np.ndarray or type(zinputs) != np.ndarray:
            raise ValueError("The input must be numpy arrays.")

        # This code snippet is checking if the dimensions of the `weights` and `zinputs` arrays are
        # 1-dimensional. If either of them is not 1-dimensional, it raises a `ValueError` with the message
        # "The input must be numpy arrays of 1D."
        if weights.ndim != 1 or zinputs.ndim != 1:
            raise ValueError("The input must be numpy arrays of 1D.")


        # (int) number used for control of dimensions.
        self.size_input : int         = np.size(zinputs)
        self.weights    : np.ndarray  = weights.T
        self.zinputs    : np.ndarray  = zinputs
        # init the ekf for the neuron
        self.ekf = ekf.ekf(self.weights, self.zinputs)
        # init the prediction as 0.
        self.prediction = 0

    def set_ekf(self, P_init : float=0.001, Q_init : float=0.001, R_init : float =0.001, FO : float =1):
        """
        Init the Extended Kalman Filter.
        Inputs
        :param  : P_init  : initial value for P matrix in the EKF.
        :param  : Q_init  : initial value for Q matrix in the EKF.
        :param  : R_init  : initial value for R in the EKF.
        :param  : FO      : Forgetting factor FO used to adjust the reaction to sudden changes in the forecast, if FO = 1 the EKFFO results must be equal to normal EKF.
        """
        self.ekf.init_filter(self.size_input, P_init, Q_init, R_init, FO)

    def update(self, new_zinputs: list, measure_signal):
        """
        Update the Extended Kalman Filter value in k-th
        Inputs
        :param  : new_zinputs     : The measurements of the inputs in the step k-th.
        :param  : measure_signal  : The value to predict. Used for calculate the error.
        """
        if type(new_zinputs) is list:
            new_zinputs: np.ndarray = np.array(new_zinputs).flatten()

        # Probe if W and Z has not the same number of elements in array.
        if np.size(new_zinputs) != np.size(self.weights):
            # The dimensions of weights and zinputs are diferents, raise an error. the size of W and Z need to be equals.
            raise ValueError("The dimensions of the vectors must be equal; size(weights) != size(zinputs)")
        

        if type(new_zinputs) != np.ndarray:
            raise ValueError("The input must be numpy arrays.")

        if new_zinputs.ndim != 1:
            raise ValueError("The input must be numpy arrays of 1D.")
        
        # check if are the same dimension from the (k - 1) iteration.
        # The size of the new z inputs dimensions, are different in this iteration.
        if np.size(new_zinputs) != np.size(self.zinputs):
            raise ValueError("The dimensions of the vector Z are diferent in this iteration.")
        
        # Update the error by passing the measure value, and the previus value of prediction.
        self.ekf.set_error(measure_signal, self.predict())      
        self.zinputs = new_zinputs
        self.weights = self.ekf.update_weights(self.zinputs)  
            
    
    def predict(self):
        """
        Calculate the value of the neuron in (k + 1)
        Return:
        :param : prediction : Numpy vectorial dot product of equation Z * W.
        """
        self.zinputs * self.weights
        self.prediction = np.dot(self.zinputs, self.weights)
        return self.prediction