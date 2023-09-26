# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import os
import inspect
from torchvision import datasets

# Gather MNIST 
train_set_tensor = datasets.MNIST('./data', train=True, download=True)
test_set_tensor = datasets.MNIST('./data', train=False, download=True)
# Convert to numpy
raw_train_set = train_set_tensor.data.numpy()
raw_test_set = test_set_tensor.data.numpy()
# Normalize 
raw_train_set = raw_train_set / 255.0
raw_test_set = raw_test_set / 255.0
# Reshape
raw_train_set = raw_train_set.reshape(-1, 784)
raw_test_set = raw_test_set.reshape(-1, 784)

np.random.seed(42)

def preview(image):
    plt.imshow(image, cmap='gray') 
    plt.axis('off') 
    plt.show()

preview(raw_test_set[0].reshape(28, 28))
# %%
from abc import ABC, abstractmethod
import numpy as np
import inspect

class LossFunctions:

    @staticmethod 
    def MSE(input_batch, predicted):
        
        loss = np.mean(np.square(input_batch - predicted))
        gradient = predicted - input_batch / np.prod(predicted.shape[:-1])

        return loss, gradient
    
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
    def update(self, learning_rate, batch_size):
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
    
    def update(self, learning_rate, batch_size):
        return
    
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
    
    def update(self, learning_rate, batch_size):
        pass

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
    
    def update(self, learning_rate, batch_size):
        pass

class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.zeros(output_size).reshape(1, output_size) - 0.5

    def forward(self, input_batch):
        self.input_batch = input_batch
        return np.dot(self.input_batch, self.weights) + self.biases

    def backward(self, error_batch):
        self.error_batch = error_batch
        self.weights_error = np.dot(self.input_batch.T, error_batch)
        input_error = np.dot(error_batch, self.weights.T)
        
        return input_error
    
    def update(self, learning_rate, batch_size):
        self.weights -= learning_rate * (np.sum(self.weights_error, axis=0) / batch_size)
        self.biases -= learning_rate * (np.sum(self.error_batch, axis = 0) / batch_size)
        


# %%
class AutoEncoder:
    
    def __init__ (
        self, 
        encode_layers,
        decode_layers,
        epochs = 100,
        learning_rate = 0.1,
        loss_function = LossFunctions.MSE,
        optimizer = 'SGD'
    ):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.encoder = encode_layers
        self.decoder = decode_layers
        self.loss_graph = {}
    
    def forward(self, input_batch):
        # Encode
        for layer in self.encoder:
            input_batch = layer.forward(input_batch)
        # Decode
        for layer in self.decoder:
            input_batch = layer.forward(input_batch)

        return input_batch
    
    def backward(self, error_batch):

        for layer in reversed(self.decoder):
            error_batch = layer.backward(error_batch)

        for layer in reversed(self.encoder):
            error_batch = layer.backward(error_batch)
    
    def train(self, train_set, batch_size = 1):
        if self.optimizer == 'SGD':
            self.train_sgd(train_set)
        
        if self.optimizer == 'Batch':
            self.train_batch(train_set)

        if self.optimizer == 'MiniBatch':
            if batch_size == 1:
                print("Warning: You are doinog MiniBatch with a batch_size of 1!")
            self.train_minibatch(train_set, batch_size)

        print(f"Error: No optimizer matched {self.optimizer}") 

    def train_sgd(self, train_set):
        epochs_time = 0
        epochs_loss = 0
        self.batch_size = 1
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_start = time.time()
            for input_batch in train_set:
                # input_batch = train_set[random.randint(0, len(train_set) - 1)]
                predicted = self.forward(input_batch.reshape(1, -1))
                loss, error_batch = self.loss_function(input_batch, predicted)
                epoch_loss += loss

                self.backward(error_batch)

                self.update_all_layers()

            self.loss_graph[epoch] = epoch_loss

            epochs_loss += epoch_loss 
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
            
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Avg Loss {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")
            self.display()

    def update_all_layers(self):
        for layer in self.decoder + self.encoder:
            layer.update(self.learning_rate, self.batch_size)

    def train_batch(self, train_batch):
        epochs_loss = 0
        epochs_time = 0
        self.batch_size = len(train_batch)
        for epoch in range(self.epochs):
            epoch_start = time.time()
            predicted = self.forward(train_batch)
            loss, error_batch = self.loss_function(train_batch, predicted)

            self.backward(error_batch)
            
            self.update_all_layers()

            self.loss_graph[epoch] = loss
            epochs_loss += loss 
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
        
            print(f"Epoch {epoch}, Loss: {loss:.4f} Avg Loss {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")

    def train_minibatch(self, train_set, batch_size):
        epochs_loss = 0
        epochs_time = 0
        self.batch_size = batch_size
        batches = [train_set[i:i+batch_size] for i in range(0, train_set.shape[0], batch_size)]
        for epoch in range(self.epochs):
            epoch_loss = 0
            batches = random.shuffle(batches)

            for train_batch in batches:
                epoch_start = time.time()
                predicted = self.forward(train_batch)
                
                loss, error_batch = self.loss_function(train_batch, predicted)

                self.backward(error_batch)

                for layer in self.decoder + self.encoder:
                    layer.update(self.learning_rate, self.batch_size)

                epoch_loss += loss 
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
            
            self.loss_graph[epoch] = epoch_loss
            epochs_loss += epoch_loss

            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Avg Loss {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")
            if epoch != 0:
                self.display()

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
autoencoder = AutoEncoder(
    encode_layers = [
        LinearLayer(784, 256),
        Sigmoid(),
        LinearLayer(256, 32),
        Sigmoid(),
    ], 
    decode_layers = [
        LinearLayer(32, 256),
        Sigmoid(),
        LinearLayer(256, 784),
        Sigmoid()
    ], 
    optimizer = 'SGD')

autoencoder.train(raw_train_set)

# %%

def test_helper(autoencoder: AutoEncoder, start_point, end_point):
    if start_point > end_point:
        print("Start point must be greater than end point!")
        return

    for test in range(start_point, end_point + 1):
        forward_result = autoencoder.forward(raw_test_set[test].reshape(1, 784))
        preview(raw_test_set[test].reshape(28, 28))
        preview(forward_result.reshape(28, 28))

def test(start_point, end_point):
    test_helper(autoencoder, start_point, end_point)

# %%
class tester:

    def __init__ (self):
        self.autoencoder = None

    def test_helper(self, autoencoder: AutoEncoder, start_point, end_point):
        if start_point > end_point:
            print("Start point must be greater than end point!")
            return

        for test in range(start_point, end_point + 1):
            forward_result = autoencoder.forward(raw_test_set[test].reshape(1, 784))
            preview(raw_test_set[test].reshape(28, 28))
            preview(forward_result.reshape(28, 28))
    
    def test(self, start_point, end_point):
        test_helper(autoencoder, start_point, end_point)

    def load_autoencoder(self, dir):

        try:
            if type(dir) == type(AutoEncoder):
                self.autoencoder = dir
        except:
            pass

        try:
            if os.path.exists(dir):
                with open(dir, 'rb') as file:
                    self.autoencoder = pickle.load(file)
        except:
            pass
        
        print("Loading autoencoder failed!")
            
# %%
