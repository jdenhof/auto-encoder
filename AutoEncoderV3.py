# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import os
import inspect
from torchvision import datasets

train_set_tensor = datasets.MNIST('./data', train=True, download=True)
test_set_tensor = datasets.MNIST('./data', train=False, download=True)

train_set = train_set_tensor.data.numpy()
test_set = test_set_tensor.data.numpy()

train_set = train_set / 255.0
test_set = test_set / 255.0

train_set = train_set.reshape(-1, 784)
test_set = test_set.reshape(-1, 784)
np.random.seed(42)

def preview(image):
    plt.imshow(image, cmap='gray') 
    plt.axis('off') 
    plt.show()

preview(test_set[0].reshape(28, 28))
# %%
from abc import ABC, abstractmethod
import numpy as np
import inspect

class LossFunctions:

    @staticmethod 
    def MSE(input_, predicted):
        
        loss = np.mean(np.square(input_ - predicted))
        gradient = predicted - input_

        return loss, gradient
    
        
class Layer(ABC):

    def __init__ (self):
        pass

    @abstractmethod
    def forward(self, input_):
        pass

    @abstractmethod
    def backward(self, cost_gradient):
        pass

    @abstractmethod
    def update(self, learning_rate):
        pass

class Sigmoid(Layer):

    def __init__ (self):
        self.result = None

    def forward(self, input_):
        self.result = 1 / (1 + np.exp(-input_))
        return self.result
    
    def backward(self, cost_gradient):
        if type(self.result) == type(None):
            print("Error: result is none")

        return cost_gradient * self.result * (1 - self.result)
    
    def update(self, learning_rate):
        return
    
class Relu(Layer):

    def __init__ (self):
        self.result = None

    def forward(self, input_):
        self.result = np.maximum(0, input_)
        return self.result

    def backward(self, cost_gradient):
        if type(self.result) == type(None):
            print("Error: result is none")

        return cost_gradient * np.where(self.result > 0, 1, 0)
    
    def update(self, learning_rate):
        pass

class Tanh(Layer):

    def __init__(self):
        self.result = None

    def forward(self, input_):
        self.result = np.tanh(input_)
        return self.result

    def backward(self, cost_gradient):
        if type(self.error) == type(None):
            print("Error: result is none")

        return cost_gradient * (1 - self.result ** 2)
    
    def update(self, learning_rate):
        pass

class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.zeros(output_size).reshape(1, output_size) - 0.5

    def forward(self, input_data):
        self.input_ = input_data
        return np.dot(self.input_, self.weights) + self.biases

    def backward(self, error):

        if type(self.input_) == type(None):
            print("Error: result is none")

        self.error = error
        self.weight_error = np.dot(self.input_.T, error)
        input_error = np.dot(error, self.weights.T)
        
        return input_error
    
    def update(self, learning_rate):

        self.weights -= learning_rate * self.weight_error
        self.biases -= learning_rate * self.error
        

# %%
class AutoEncoder:
    
    def __init__ (
        self, 
        encode_layers,
        decode_layers,
        epochs = 20,
        learning_rate = 0.01,
        loss_function = LossFunctions.MSE
    ):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.encoder = encode_layers
        self.decoder = decode_layers
        self.loss_graph = {}
    
    def forward(self, input_data):
        for layer in self.encoder:
            input_data = layer.forward(input_data)

        for layer in self.decoder:
            input_data = layer.forward(input_data)

        return input_data
    
    def backward(self, gradient):

        for layer in reversed(self.decoder):
            gradient = layer.backward(gradient)
            layer.update(self.learning_rate)
        
        for layer in reversed(self.encoder):
            gradient = layer.backward(gradient)
            layer.update(self.learning_rate)

    def epochtime(self, epoch, epoch_start, epochs_time):

        epoch_total = time.time() - epoch_start
        epochs_time += epoch_total
        avg_time = epochs_time / (epoch + 1)
        est_time = (self.epochs - (epoch + 1)) * avg_time # seconds total
        hours_in_secs = (60*60)
        min_in_secs = 60

        hours = int(est_time / hours_in_secs)
        mins = int((est_time - hours_in_secs*hours) / min_in_secs)
        seconds = int((est_time - hours_in_secs* hours - min_in_secs*mins))

        return hours, mins, seconds, epoch_total, avg_time, epochs_time

    def train(self, train_set):
        epochs_time = 0
        epochs_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_start = time.time()
            for input_ in train_set:

                predicted = self.forward(input_.reshape(1, -1))
                loss, gradient = self.loss_function(input_, predicted)
                epoch_loss += loss

                self.backward(gradient)

            self.loss_graph[epoch] = epoch_loss

            epochs_loss += epoch_loss / len(train_set)
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
            
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Avg Loss {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")

            if (epoch + 1 % 10 == 0):
                self.display()

        self.save()

    def save(self):
        x = 0
        def dir(x):
            return f'autoencoder_{self.learning_rate}_{self.epochs}_{x}.pkl'

        while(os.path.exists(dir(x))):
            x += 1
        
        with open(dir(x), 'wb') as file:
            pickle.dump(self, file)

    def display(self):
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        plt.plot(self.loss_graph.keys(), self.loss_graph.values(), marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Per Epoch')
        plt.grid(True)
        plt.show()

# %%

encoder = [
    LinearLayer(784, 256),
    Sigmoid(),
    LinearLayer(256, 32),
    Sigmoid(),
]

decoder = [
    LinearLayer(32, 256),
    Sigmoid(),
    LinearLayer(256, 784),
    Sigmoid()
]

autoencoder = AutoEncoder(encode_layers = encoder, decode_layers = decoder)
autoencoder.train(train_set)

# %%
