from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__ (self):
        pass

    @abstractmethod
    def forward(self, input_batch):
        pass

    @abstractmethod
    def backward(self, error_batch):
        pass

    @abstractmethod
    def update(self, learning_rate, opt):
        pass



class Sigmoid(Layer):
    def __init__ (self):
        self.input_batch = None

    def forward(self, input_batch):
        self.input_batch = 1 / (1 + np.exp(-input_batch))
        return self.input_batch
    
    def backward(self, error_batch):
        if type(self.input_batch) == type(None):
            print("Error: result is none")

        return error_batch * self.input_batch * (1 - self.input_batch)



class Relu(Layer):
    def __init__ (self):
        self.input_batch = None

    def forward(self, input_batch):
        self.input_batch = np.maximum(0, input_batch)
        return self.input_batch

    def backward(self, cost_gradient):
        if type(self.input_batch) == type(None):
            print("Error: result is none")

        return cost_gradient * np.where(self.input_batch > 0, 1, 0)



class Tanh(Layer):
    def __init__(self):
        self.input_batch = None

    def forward(self, input_batch):
        self.input_batch = np.tanh(input_batch)
        return self.input_batch

    def backward(self, error_batch):
        if type(self.input_batch) == type(None):
            print("Error: result is none")

        return error_batch * (1 - self.input_batch ** 2)



class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.zeros((1, output_size)) - 0.5
        self.m_w = np.zeros((input_size, output_size))
        self.v_w = np.zeros((input_size, output_size))
        self.m_b = np.zeros((1, output_size))
        self.v_b = np.zeros((1, output_size))
        self.input_size = input_size
        self.output_size = output_size
        self.step = 0

    def forward(self, input_batch):
        self.input_batch = input_batch
        return np.dot(self.input_batch, self.weights) + self.biases

    def backward(self, error_batch):
        batch_size = error_batch.shape[0]
        self.biases_error = (np.sum(error_batch, axis=0) / batch_size)
        self.weights_error = np.sum(np.einsum('ij,ik->ijk', self.input_batch, error_batch), axis = 0) / batch_size

        return np.dot(error_batch, self.weights.T)
    
    def update(self, learning_rate, optimizer):

        if optimizer == 'ADAM':
            self.update_adam( learning_rate )
        else:
            self.weights -= learning_rate * self.weights_error
            self.biases -= learning_rate * self.biases_error

    def update_adam( self, learning_rate ):
        beta1 = 0.9 # TODO: FIX
        beta2 = 0.999 # TODO: FIX
        eps = 1e-8 # TODO: FIX

        self.step += 1

        self.m_w = beta1 * self.m_w + (1 - beta1) * self.weights_error
        self.v_w = beta2 * self.v_w + (1 - beta2) * np.power(self.weights_error, 2)
        mhat_w = self.m_w / (1 - np.power(beta1, self.step))
        vhat_w = self.v_w / (1 - np.power(beta2, self.step))

        self.m_b = beta1 * self.m_b + (1 - beta1) * self.biases_error
        self.v_b = beta2 * self.v_b + (1 - beta2) * (np.power(self.biases_error, 2))
        mhat_b = self.m_b / (1 - np.power(beta1, self.step))
        vhat_b = self.v_b / (1 - np.power(beta2, self.step))

        self.weights -= learning_rate * mhat_w / (np.sqrt(vhat_w) + eps)
        self.biases -= learning_rate * mhat_b / (np.sqrt(vhat_b) + eps)