from collections.abc import Iterable
import numpy as np
import inspect
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import os
import inspect
from torchvision import datasets

class LossFunctions:

    @staticmethod 
    def MSE(input_batch, predicted):

        loss = np.square(input_batch - predicted).mean()
        gradient = (predicted - input_batch) # / np.prod(predicted.shape[:-1])

        return loss, gradient

class Layer:

    def __init__ (self):
        self.shape = None
        raise NotImplementedError()
    
    def __call__(self, input_batch):
        raise NotImplementedError()

    def backward(self, error_batch):
        raise NotImplementedError()

    def update(self, learning_rate, opt):
        raise NotImplementedError()

    def _shape(self):
        raise NotImplementedError()

class Sigmoid(Layer):

    def __init__ (self):
        self.input_batch = None
        self.shape = self._shape()

    def __call__(self, input_batch):
        self.input_batch = 1 / (1 + np.exp(-input_batch))
        return self.input_batch
    
    def backward(self, error_batch):
        if type(self.input_batch) == type(None):
            print("Error: result is none")

        return error_batch * self.input_batch * (1 - self.input_batch)
    
    def update(self, learning_rate, opt):
        return

    def _shape(self):
        return None
    
class Relu(Layer):

    def __init__ (self):
        super().__init__()
        self.input_batch = None
        self.shape = self._shape()

    def __call__(self, input_batch):
        self.input_batch = np.maximum(0, input_batch)
        return self.input_batch

    def backward(self, cost_gradient):
        if type(self.input_batch) == type(None):
            print("Error: result is none")

        return cost_gradient * np.where(self.input_batch > 0, 1, 0)
    
    def update(self, learning_rate, opt):
        pass

    def _shape(self):
        return None

class Tanh(Layer):

    def __init__(self):
        super().__init__()
        self.input_batch = None
        self.shape = self._shape()

    def __call__(self, input_batch):
        self.input_batch = np.tanh(input_batch)
        return self.input_batch

    def backward(self, error_batch):
        if type(self.input_batch) == type(None):
            print("Error: result is none")

        return error_batch * (1 - self.input_batch ** 2)
    
    def update(self, learning_rate, opt):
        pass

    def _shape(self):
        return None

class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.zeros((1, output_size), ) - 0.5
        self.m_w = np.zeros((input_size, output_size))
        self.v_w = np.zeros((input_size, output_size))
        self.m_b = np.zeros((1, output_size)) 
        self.v_b = np.zeros((1, output_size)) 
        self.input_size = input_size
        self.output_size = output_size
        self.step = 0
        self.shape = self._shape()

    def __call__(self, input_batch):
        self.input_batch = input_batch
        return np.dot(self.input_batch, self.weights) + self.biases

    def backward(self, error_batch):
        batch_size = error_batch.shape[0]
        self.biases_error = (np.sum(error_batch, axis=0) / batch_size)
        self.weights_error = np.sum(np.einsum('ij,ik->ijk', self.input_batch, error_batch), axis=0) / batch_size

        return np.dot(error_batch, self.weights.T)
    
    def update(self, learning_rate, optimizer):     

        if optimizer == 'ADAM':
            self.update_adam(learning_rate)
            return

        self.weights -= learning_rate * self.weights_error
        self.biases -= learning_rate * self.biases_error

    def update_adam(self, learning_rate):
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

    def _shape(self):
        return (self.input_size, self.output_size)

    def __str__(self):
        return str(self.shape)

class Sequential(Layer):

    def __init__(self, *items: Layer):
        self.items = items
        self.shape = self._shape()

    def __call__(self, input):
        for layer in self.items:
            input = layer(input)
            
        return input

    def backward(self, input):
        for layer in reversed(self.items):
            input = layer.backward(input)

        return input

    def update(self, learning_rate, optimizer):
        for layer in self.items:
            layer.update(learning_rate, optimizer)

    def _shape(self):
        shape = []
        for item in self.items:
            item_shape = item.shape
            if item_shape is not None:
                shape.append(item_shape)
    
        return shape

    def __str__(self):
        string = "\n"
        for item in self.shape:
            string += str(item) + "\n"
        return string
    
class Parrallel(Layer):
    
    def __init__(self, *args: Layer):
        self.items = args
        self.size = len(args)
        self.shape = self._shape()

    def __call__(self, input):
        return [item(input) for item in self.items]
    
    def backward(self, input):
        return [item.backward(input) for item in self.items]
    
    def update(self, learning_rate, optimizer):
        for item in self.items:
            item.update(learning_rate, optimizer)

    def _shape(self):
        shape = []
        for item in self.items:
            item_shape = item.shape
            if item_shape is not None:
                shape.append(item_shape)

        return shape
    
    def __str__(self):
        string = ""
        for item in self.shape:
            string += str(item) + "\t"
        return string
    
class MNIST:

    @staticmethod

    def get_data(dir):
        """
        Defaults to not download data, set dir to the directory to store MNSIT
        """

        train_set_tensor = datasets.MNIST(dir, train=True, download=True)
        test_set_tensor = datasets.MNIST(dir, train=False, download=True)

        # Convert to numpy
        raw_train_set = train_set_tensor.data.numpy()
        raw_test_set = test_set_tensor.data.numpy()
        # Normalize 
        raw_train_set = raw_train_set / 255.0
        raw_test_set = raw_test_set / 255.0
        # Reshape
        raw_train_set = raw_train_set.reshape(-1, 784)
        raw_test_set = raw_test_set.reshape(-1, 784)

        return raw_test_set, raw_train_set

    @staticmethod
    def preview(image):
        plt.imshow(image.reshape(28, 28), cmap='gray') 
        plt.axis('off') 
        plt.show()
