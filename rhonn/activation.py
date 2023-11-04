
from abc import ABC, abstractmethod
import math

# The class activation is an abstract base class that defines a method activate().
class activation(ABC):
    @abstractmethod
    def activate(self, c):
        pass

class soft_sigmoid(activation):
    def activate(self, c, B=0.83):
        """
        activation function soft sigmoidal.
        Inputs
        :param  : c    : Value of the signal in k-th
        :param  : B    : Beta used for the adjustment of the sigmoidal.

        Return:
        Result of the operation 1/(1 + e^(-B * C)) 
        """
        return 1 / (1 + math.exp(-B * c))