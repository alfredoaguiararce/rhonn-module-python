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
import math
import ekf

class rhonn:
    def __init__(self, weights, zinputs):
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
        self.ekf = ekf.ekf(self.weights, self.zinputs)
        # init the prediction as 0.
        self.prediction = 0

    def set_ekf(self, P_init  =0.001, Q_init  =0.001, R_init  =0.001, FO=1):
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