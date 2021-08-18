# -*- coding: utf-8 -*-
"""
Created for: Instituto Tecnologico de Tepic : Recurrent High Order Neural Networks (RHONN).
@author: Alfredo Aguiar Arce. (www.alfredoagrar.com)

Pruebas unitarias
"""
import unittest
from rhonn import rhonn
from rhonn import activation
import numpy as np


class TestRHONN(unittest.TestCase):

    def test_neuron(self):
        """ Probamos que el resultado inicial que arroja la neurona sea correcto."""
        # crear una lista 1-D (Horizontal, Entradas).
        Z = [1, 2, 3]
        # crear una lista 1-D (Vertical, Pesos de la red).
        W = [10, 20, 30]
        # Inicializamos la neurona, y obtenemos el valor que toma dado W * Z
        # X(k) = W * Z
        result = rhonn(W, Z).predict()
        # Comprobamos el resultado 
        self.assertEqual(result, 140)

    def test_vector_dimensions(self):
        """ Probamos que si se ingresan vectores de diferente dimension, el programa pueda detectar el error. """
        # crear una lista 1-D (Horizontal, Entradas). 
        Z = [1, 2, 3, 4, 5]
        # crear una lista 1-D (Vertical, Pesos de la red).
        W = [10, 20, 30]
        # Notemos que las dimensiones de Z y W son diferentes.
        try:
            neuron = rhonn(W, Z)
        except ValueError as e:
            # Comprobamos que efectivamente hay un error en las dimensiones.
            self.assertEqual(type(e), ValueError)
        else:
            self.fail('El error no fue lanzado.')

    def test_init_ekf(self):
        # crear una lista 1-D (Horizontal, Entradas). 
        Z = [0, 0]
        # crear una lista 1-D (Vertical, Pesos de la red).
        W = [0, 0]

        neuron = rhonn(W, Z)
        neuron.set_ekf(1,1,1)

        # Acesses to the internal values of the internal EKF
        P = neuron.ekf.get_P()
        Q = neuron.ekf.get_Q()
        R = neuron.ekf.get_R()

        P_test = [[1, 0], 
                  [0, 1]]

        Q_test = [[1, 0], 
                  [0, 1]]
        
        R_test = [1]
                  
        self.assertTrue((P == P_test).all())
        self.assertTrue((Q == Q_test).all())
        self.assertTrue((R == R).all())


    def test_activation_sigmoid(self):
        """ Probamos el resultado de la funcion de activacion S(*)"""
        senal = 1
        result = activation.soft_sigmoid(senal)
        self.assertEqual(result, 0.6963549298238342)


if __name__ == '__main__':
    # Si este  modulo se ejecuta directamente se correran todos los test en TestRHONN.
    unittest.main()