import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import sys 
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

try:
    import NN
except:
    print("Failed to import module NN")

class Model(NN.Sequential):
    
    def __init__ (
        self,
        *args,
        epochs = 100,
        learning_rate = 0.01,
        loss_function = NN.LossFunctions.MSE,
        optimizer = 'ADAM'
    ):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.loss_graph = {}

        if (len(args) == 0):
            layers = NN.Sequential(
                NN.Sequential(
                    NN.LinearLayer(784, 256),
                    NN.Sigmoid(),
                    NN.LinearLayer(256, 32),
                    NN.Sigmoid(),
                ),
                NN.Sequential(
                    NN.LinearLayer(32, 256),
                    NN.Sigmoid(),
                    NN.LinearLayer(256, 784),
                    NN.Sigmoid()
                )
            )

        super().__init__(layers)

    def __call__(self, input):
        predicted = super().__call__(input)

        loss, gradient = self.loss_function(input, predicted)

        self.backward(gradient)

        return loss, predicted
    
    def train(self, train_set, batch_size = 1):
        trained = False
        if self.optimizer == 'SGD':
            self.train_sgd(train_set)
            trained = True
        
        if self.optimizer == 'Batch':
            self.train_batch(train_set)
            trained = True

        if self.optimizer == 'MiniBatch':
            if batch_size == 1:
                print("Warning: Default MiniBatch of size 32 in use.")
                batch_size = 32
            self.train_minibatch(train_set, batch_size)
            trained = True

        if self.optimizer == 'ADAM':
            if batch_size == 1:
                print("Warning: Default MiniBatch of size 32 in use.")
            self.train_adam(train_set, batch_size)
            trained = True

        if not trained:
            print(f"Error: No optimizer matched {self.optimizer}")

    def train_sgd(self, train_set):
        print("Training with Stochastic Gradient Descent")
        epochs_time = 0
        epochs_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_start = time.time()
            np.random.shuffle(train_set)

            for input_batch in train_set:
                loss, _ = self(input_batch.reshape(1, -1))
                self.update(self.learning_rate, self.optimizer)
                epoch_loss += loss

            self.loss_graph[epoch] = epoch_loss

            epochs_loss += epoch_loss 
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
            
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Avg Loss/Sample: {epoch_loss / train_set.shape[0]} Avg Loss: {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")
            self.display()

    def train_adam(self, train_set, batch_size):
        print("Training with Adam")
        epochs_time = 0
        epochs_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_start = time.time()
            np.random.shuffle(train_set)
            batches = np.split(train_set, len(train_set) // batch_size)
            for train_batch in batches:
                loss, _ = self(train_batch)
                epoch_loss += loss
                self.update(self.learning_rate, self.optimizer)

            self.loss_graph[epoch] = epoch_loss

            epochs_loss += epoch_loss 
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
            
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Avg Loss/Sample: {epoch_loss / train_set.shape[0]} Avg Loss: {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")
            self.display()

    def train_batch(self, train_batch):
        print("Training Gradient Descent")
        epochs_loss = 0
        epochs_time = 0
        for epoch in range(self.epochs):
            epoch_start = time.time()

            loss, _ = self(train_batch)
            self.update(self.learning_rate, self.optimizer)
            epochs_loss += loss 
            hours, mins, seconds, epoch_total, avg_time, epochs_time = self.epochtime(epoch, epoch_start, epochs_time)
        
            print(f"Epoch {epoch}, Loss: {loss:.4f} Avg Loss: {epochs_loss / (epoch + 1):.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")

    def train_minibatch(self, train_set, batch_size):
        print(f"Training with MiniBatch [{batch_size}]")
        epochs_loss = 0
        epochs_time = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            batches = [train_set[i:i+batch_size] for i in range(0, train_set.shape[0], batch_size)]
            np.random.shuffle(batches)
            epoch_start = time.time()

            for train_batch in batches:
                loss, _ = self(train_batch) 
                epoch_loss += loss
                self.update(self.learning_rate, self.optimizer)

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
            return f'autoencoder_{self.learning_rate}_{self.epochs}_{x}_{self.optimizer}.pkl'

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


class Tester:

    def __init__ ( self, vae: Model, test_set: [] ):
        self.vae = vae
        self.test_set = test_set

    def __call__ (self, *tests):
        for test in tests:
            test = self.test_set[test]
            NN.MNIST.preview(test)
            encoded, mu, log_var, decoded = self.vae.forward(test)
            NN.MNIST.preview(decoded)