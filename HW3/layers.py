import numpy as np


class Dense:
    """
    Fully connected Dense layer
    """
    def __init__(self, input_D, output_D):
        """
        :param input_D: Input dimension
        :param output_D: Output dimension
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
        self.prob = None

    def forward(self, X, Y):
        # One hot encode Y
        NUM_CLASSES = 10
        Y2 = Y.ravel()
        self.Y_onehot = np.zeros((len(Y2), NUM_CLASSES))
        self.Y_onehot[np.arange(len(Y2)), Y2] = 1

        # Compute Softmax function
        X_normalized = X - np.amax(X, axis=1, keepdims=True)
        X_exp = np.exp(X_normalized)
        sum_exp = np.sum(X_exp, axis=1, keepdims=True)
        self.prob = X_exp / sum_exp

        # Compute loss function
        forward_output = - np.sum(self.Y_onehot * (X_normalized - np.log(sum_exp))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = (self.prob - self.Y_onehot) / X.shape[0]
        return backward_output
