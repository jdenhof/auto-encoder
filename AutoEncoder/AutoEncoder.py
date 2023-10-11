#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from torchvision import datasets, transforms

train_set_tensor = datasets.MNIST('./data', train=True, download=True)
test_set_tensor = datasets.MNIST('./data', train=False, download=True)

train_set = train_set_tensor.data.numpy()
test_set = test_set_tensor.data.numpy()

train_set = train_set / 255.0
test_set = test_set / 255.0

train_set = train_set.reshape(-1, 784)
test_set = test_set.reshape(-1, 784)


# In[3]:


def preview(image):
    plt.imshow(image, cmap='gray') 
    plt.axis('off') 
    plt.show()


# In[4]:


input_size = 28*28
hidden_size = int(input_size / 2)
latent_size = 32
preview(test_set[0].reshape(28, 28))
np.random.seed(42)


# In[5]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
















# In[7]:

learning_rate = 0.01
epochs = 80

input_size = 28*28
hidden_size = 256
latent_size = 128
# y -> z1 -> z2 -> z3 -> y_pred

W1 = np.random.randn(input_size, hidden_size)  # Weights for the hidden layer
b1 = np.zeros((1, hidden_size))  # Biases for the hidden layer
W2 = np.random.randn(hidden_size, latent_size)  # Weights for the output layer
b2 = np.zeros((1, latent_size))
W3 = np.random.randn(latent_size, hidden_size)  # Weights for the hidden layer
b3 = np.zeros((1, hidden_size))  # Biases for the hidden layer
W4 = np.random.randn(hidden_size, input_size)  # Weights for the output layer
b4 = np.zeros((1, input_size))

def forward(y):
    z1 = np.dot(y, W1) + b1 # input_size -> hidden_size
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 # hidden_size -> latent_size
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3 # latent_size -> hidden_size
    a3 = sigmoid(z3)  
    y_pred = np.dot(a3, W4) + b4 # hidden_size -> input_size
    return z1, a1, z2, a2, z3, a3, y_pred

def backward(z1, a1, z2, a2, z3, a3, y_pred, y):
    dLdz4 = y_pred - y

for epoch in range(epochs):
    epoch_loss = 0
    for y in train_set:
        # Forward propagation
        backward(forward(y), y)
    
        loss = np.mean(0.5 * np.power(y - y_pred, 2)) # Mean Square Loss function
        epoch_loss += loss
        dz4 = y_pred - y  # Assuming a4 is the output of your network
        dW4 = a3.T.dot(dz4)  # Gradient of loss w.r.t. W4
        db4 = np.sum(dz4, axis=0)  # Gradient of loss w.r.t. b4
            
        dz3 = dz4.dot(W4.T) * a3 * (1 - a3)  # Gradient of loss w.r.t. z3 using sigmoid derivative
        dW3 = a2.T.dot(dz3)  # Gradient of loss w.r.t. W3
        db3 = np.sum(dz3, axis=0)  # Gradient of loss w.r.t. b3
        
        dz2 = dz3.dot(W3.T) * a2 * (1 - a2) # Gradient of loss w.r.t. z2 using ReLU derivative
        dW2 = a1.T.dot(dz2)  # Gradient of loss w.r.t. W2
        db2 = np.sum(dz2, axis=0)  # Gradient of loss w.r.t. b2
        
        dz1 = dz2.dot(W2.T) * a1 * (1 - a1)  # Gradient of loss w.r.t. z1 using sigmoid derivative
        dW1 = dz1  # Gradient of loss w.r.t. W1
        db1 = np.sum(dz1, axis=0)  # Gradient of loss w.r.t. b1
        
        # Update weights and biases for each layer
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        
        W3 = W3 - learning_rate * dW3
        b3 = b3 - learning_rate * db3
        
        W4 = W4 - learning_rate * dW4
        b4 = b4 - learning_rate * db4
        
    print(f"Epoch {epoch}, Loss: {epoch_loss}")


# In[8]:

import time
# Define the batch size (number of samples in each mini-batch)
batch_size = 32  # You can adjust this value based on your dataset and available memory

# Split your training data into mini-batches
num_samples = len(train_set)
num_batches = num_samples // batch_size

learning_rate = 0.01
epochs = 80

input_size = 28*28
hidden_size = 256
latent_size = 128

W1 = np.random.randn(input_size, hidden_size)  # Weights for the hidden layer
b1 = np.zeros((1, hidden_size))  # Biases for the hidden layer
W2 = np.random.randn(hidden_size, latent_size)  # Weights for the output layer
b2 = np.zeros((1, latent_size))
W3 = np.random.randn(latent_size, hidden_size)  # Weights for the hidden layer
b3 = np.zeros((1, hidden_size))  # Biases for the hidden layer
W4 = np.random.randn(hidden_size, input_size)  # Weights for the output layer
b4 = np.zeros((1, input_size))

def forward(y):
    z1 = np.dot(y, W1) + b1 # input_size -> hidden_size
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 # hidden_size -> latent_size
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3 # latent_size -> hidden_size
    a3 = sigmoid(z3)  
    z4 = np.dot(a3, W4) + b4 # hidden_size -> input_size
    return z1, a1, z2, a2, z3, a3, z4

epochs_time = 0
for epoch in range(epochs):
    # Shuffle your training data at the beginning of each epoch to randomize mini-batch order
    np.random.shuffle(train_set)
    epoch_start = time.time()
    for i in range(num_batches):
        # Extract a mini-batch
        start = i * batch_size
        end = (i + 1) * batch_size
        mini_batch = train_set[start:end]
        
        # Initialize gradients for each batch
        dW1_batch, db1_batch = 0, 0
        dW2_batch, db2_batch = 0, 0
        dW3_batch, db3_batch = 0, 0
        dW4_batch, db4_batch = 0, 0
        
        # Process each sample in the mini-batch
        for y in mini_batch:
            # Forward propagation
            z1, a1, z2, a2, z3, a3, z4, y_pred = forward(y)

            dz4 = y_pred - y  # Assuming a4 is the output of your network
            dW4 = a3.T.dot(dz4)  # Gradient of loss w.r.t. W4
            db4 = np.sum(dz4, axis=0)  # Gradient of loss w.r.t. b4
                
            dz3 = dz4.dot(W4.T) * a3 * (1 - a3)  # Gradient of loss w.r.t. z3 using sigmoid derivative
            dW3 = a2.T.dot(dz3)  # Gradient of loss w.r.t. W3
            db3 = np.sum(dz3, axis=0)  # Gradient of loss w.r.t. b3
            
            dz2 = dz3.dot(W3.T) * a2 * (1 - a2)  # Gradient of loss w.r.t. z2 using ReLU derivative
            dW2 = a1.T.dot(dz2)  # Gradient of loss w.r.t. W2
            db2 = np.sum(dz2, axis=0)  # Gradient of loss w.r.t. b2
            
            dz1 = dz2.dot(W2.T) * a1 * (1 - a1)  # Gradient of loss w.r.t. z1 using sigmoid derivative
            dW1 = dz1 # Gradient of loss w.r.t. W1
            db1 = np.sum(dz1, axis=0)  # Gradient of loss w.r.t. b1

            dW1_batch += dW1
            dW2_batch += dW2
            dW3_batch += dW3
            dW4_batch += dW4

        
        W1 = W1 - (learning_rate / batch_size) * (dW1_batch / batch_size)
        b1 = b1 - (learning_rate / batch_size) * (db1_batch / batch_size)

        W2 = W2 - (learning_rate / batch_size) * (dW2_batch / batch_size)
        b2 = b2 - (learning_rate / batch_size) * (db2_batch / batch_size)

        W3 = W3 - (learning_rate / batch_size) * (dW3_batch / batch_size)
        b3 = b3 - (learning_rate / batch_size) * (db3_batch / batch_size)
        
        W4 = W4 - (learning_rate / batch_size) * (dW4_batch / batch_size)
        b4 = b4 - (learning_rate / batch_size) * (db4_batch / batch_size)
        
        
    # Calculate and print the loss at the end of each epoch
    total_loss = 0
    for y in train_set:
        _, _, _, _, _, _, _, y_pred = forward(y)
        loss = np.mean(0.5 * np.power(y - y_pred, 2))
        total_loss += loss
    avg_loss = total_loss / num_samples
    
    epoch_total = time.time() - epoch_start
    epochs_time += epoch_total
    avg_time = epochs_time / (epoch + 1)
    est_time = (epochs - (epoch + 1)) * avg_time # seconds total
    hours_in_secs = (60*60)
    min_in_secs = 60
    
    hours = int(est_time / hours_in_secs)
    mins = int((est_time - hours_in_secs*hours) / min_in_secs)
    seconds = int((est_time - hours_in_secs* hours - min_in_secs*mins))

    print(f"Epoch {epoch}, Batch Loss: {total_loss:.4f} Avg Loss {avg_loss:.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")


# In[9]:
sample = test_set[4]
preview(sample.reshape(28, 28))
_, _, _, _, _, _, result = forward(sample)
preview(result.reshape(28, 28))
    
# %%
_, _, _, _, _, _, _, y_pred = forward(y)
# %%
preview(y_pred.reshape(28, 28))
preview(y.reshape(28, 28))

# %% 









import torch
from torchvision import datasets,transforms

# specific the data path in which you would like to store the downloaded files
# here, we save it to the folder called "mnist_data"
# ToTensor() here is used to convert data type to tensor, so that can be used in network

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)

print(train_dataset)

batchSize=128

#only after packed in DataLoader, can we feed the data into the neural network iteratively
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

# %% 
# package we used to manipulate matrix
import numpy as np
# package we used for image processing
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

def imshow(img):
    npimg = img.numpy()
    #transpose: change array axis to correspond to the plt.imshow() function     
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

# load the first 16 training samples from next iteration
# [:16,:,:,:] for the 4 dimension of examples, first dimension take first 16, other dimension take all data
# arrange the image in grid
examples, _ = next(iter(train_loader))
example_show=make_grid(examples[:16,:,:,:], 4)

# then display them
imshow(example_show)

# %%
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)


# Building the encoder
class Autoencoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2):
        super(Autoencoder, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        # decoder part
        self.fc3 = nn.Linear(h_dim2, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def decoder(self, x):
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# When initialzing, it will run __init__() function as above
model = Autoencoder(num_input, num_hidden_1, num_hidden_2)

# %%

# define loss and parameters
optimizer = optim.Adam(model.parameters())
epoch = 100
# MSE loss will calculate Mean Squared Error between the inputs 
loss_function = nn.MSELoss()
epoch_time = 0

print('====Training start====')
for i in range(epoch):
    train_loss = 0
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        # prepare input data      
        inputs = torch.reshape(data,(-1, 784)) # -1 can be any value. So when reshape, it will satisfy 784 first

        # set gradient to zero
        optimizer.zero_grad()
        
        # feed inputs into model
        recon_x = model(inputs)
        
        # calculating loss 
        loss = loss_function(recon_x, inputs)
        
        # calculate gradient of each parameter
        loss.backward()
        train_loss += loss.item()
        
        # update the weight based on the gradient calculated
        optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time
    epoch_time += total_time
        
    if i%10==0:    
        print('====> Epoch: {} Average loss: {:.9f} Time: {:.2f} Avg Time: {:.2f} Estimated: {:.2f}'.format(i, train_loss, total_time, epoch_time / (i + 1),  (epoch - (i + 1))*(epoch_time / (i + 1)) / 60 ))
print('====Training finish====')
# %%
# load 16 images from testset
inputs, _ = next(iter(test_loader))
inputs_example = make_grid(inputs[:16,:,:,:],4)
imshow(inputs_example)

#convert from image to tensor
inputs=torch.reshape(inputs,(-1,784))

# get the outputs from the trained model
outputs=model(inputs)

#convert from tensor to image
outputs=torch.reshape(outputs,(-1,1,28,28))
outputs=outputs.detach().cpu()

#show the output images
outputs_example = make_grid(outputs[:16,:,:,:],4)
imshow(outputs_example)
# %%
def Autoencoder( data, epochs, learning_rate ):

    W1 = np.random.randn(input_size, hidden_size)  # Weights for the hidden layer
    b1 = np.zeros((1, hidden_size))  # Biases for the hidden layer
    W2 = np.random.randn(hidden_size, latent_size)  # Weights for the output layer
    b2 = np.zeros((1, latent_size))
    W3 = np.random.randn(latent_size, hidden_size)  # Weights for the hidden layer
    b3 = np.zeros((1, hidden_size))  # Biases for the hidden layer
    W4 = np.random.randn(hidden_size, input_size)  # Weights for the output layer
    b4 = np.zeros((1, input_size))

    z1 = np.dot(y, W1) + b1 # input_size -> hidden_size
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 # hidden_size -> latent_size
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3 # latent_size -> hidden_size
    a3 = sigmoid(z3)  
    z4 = np.dot(a3, W4) + b4 # hidden_size -> input_size
    y_pred = sigmoid(z4)

    dLdyp = y_pred - y
    
    output_error4 = dLdyp * y_pred * (1 - y_pred)
    weight_error4 = np.dot(np.transpose(a3), output_error4)
    input_error4 = np.dot(output_error4, W4.T)

    W4 -= learning_rate * weight_error4
    b4 -= learning_rate * output_error4

    output_error3 = input_error4 * a3 * (1 - a3)
    weight_error3 = np.dot(np.transpose(a2), output_error3)
    input_error3 = np.dot(output_error3, W3.T)

    W3 -= learning_rate * weight_error3
    b3 -= learning_rate * output_error3

    

    output_error2 = input_error3 * a2 * (1 - a2)
    weight_error2 = np.dot(np.transpose(a1), output_error2)
    input_error2 = np.dot(output_error2, W2.T)

    W2 -= learning_rate * weight_error2
    b2 -= learning_rate * output_error2

    output_error1 = input_error2 * a1 * (1 - a1)
    weight_error1 = np.dot(np.transpose(y), output_error1)

    W1 -= learning_rate * weight_error1
    b1 -= learning_rate * output_error1


y = test_set[0]
Autoencoder(y.reshape(1, 784))

# %%
