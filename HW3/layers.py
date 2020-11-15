import numpy as np
import pandas as pd


class Dense:
    """
    Fully connected Dense layer
    """
    def __init__(self, input_D, output_D):
        """
        :param input_D: the dimensionality of the input example/instance of the forward pass
        :param output_D: the dimensionality of the output example/instance of the forward pass
        """
        self.params = dict()
        # Xavier init https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
        self.params['W'] = np.random.normal(0, np.sqrt(2 / (input_D + output_D)), (input_D, output_D))
        self.params['b'] = np.random.normal(0, np.sqrt(2 / (input_D + output_D)), (1, output_D))
        self.params['dW'] = np.zeros((input_D, output_D))
        self.params['db'] = np.zeros((1, output_D))

    def forward(self, X):
        forward_output = X @ self.params['W'] + self.params['b']
        return forward_output

    def backward(self, X, grad):
        self.params['dW'] = np.transpose(X) @ grad
        self.params['db'] = np.transpose(grad) @ np.ones(np.size(grad, 0))
        backward_output = grad @ np.transpose(self.params['W'])
        return backward_output


class Relu:
    """
    ReLU layer
    """
    def __init__(self):
        pass

    def forward(self, X):
        forward_output = X * (X > 0).astype(float)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(1. * (X >= 0).astype(float), grad)
        return backward_output


class SoftmaxCrossEntropy:
    def __init__(self):
        self.Y_onehot = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        # One hot encode Y
        self.Y_onehot = np.array(pd.get_dummies(np.ravel(Y)), dtype="float")

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.Y_onehot, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.Y_onehot - self.prob) / X.shape[0]
        return backward_output
