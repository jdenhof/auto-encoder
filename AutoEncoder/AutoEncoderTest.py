# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import os

from torchvision import datasets    

train_set_tensor = datasets.MNIST('./data', train=True, download=True)
test_set_tensor = datasets.MNIST('./data', train=False, download=True)

train_set = train_set_tensor.data.numpy()
test_set = test_set_tensor.data.numpy()

train_set = train_set / 255.0
test_set = test_set / 255.0

train_set = train_set.reshape(-1, 784)
test_set = test_set.reshape(-1, 784)

def preview(image):
    plt.imshow(image, cmap='gray') 
    plt.axis('off') 
    plt.show()

preview(test_set[0].reshape(28, 28))

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_weights(W1, b1, W2, b2, W3, b3, W4, b4, y):
        result = forward_helper_full(W1, b1, W2, b2, W3, b3, W4, b4, y)
        return result

def forward_helper_full(W1, b1, W2, b2, W3, b3, W4, b4, y, prev = True):
    z1 = np.dot(y, W1) + b1 # input_size -> hidden_size
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 # hidden_size -> latent_size
    a2 = sigmoid(z2)
    size = int(np.sqrt(len(a2.T)))

    if prev:
        try:
            preview(a2.reshape(size, size))
        except:
            print("cannot preview latent space")

        print(f"Latent Size: {a2.shape}")
    
    z3 = np.dot(a2, W3) + b3 # latent_size -> hidden_size
    a3 = sigmoid(z3)  
    z4 = np.dot(a3, W4) + b4 # hidden_size -> input_size
    y_pred = sigmoid(z4)
    return y_pred

def test_helper(W1, b1, W2, b2, W3, b3, W4, b4, *v):
    
    for val in range(len(v) - 1):
        sample = test_set[v[val]]
        sample = sample.reshape(1, 28*28)

        preview(sample.reshape(28, 28))
    
        result = forward_weights(W1, b1, W2, b2, W3, b3, W4, b4, sample)

        preview(result.reshape(28, 28))

def load_weights(dir):
    if not os.path.exists(dir):
        print("Could not load waits")
        return None

    with open(dir, 'rb') as file:
        return pickle.load(file)

def test(*v):
    dir = v[-1]
    if dir != None:
        if os.path.exists(dir):
            W1, b1, W2, b2, W3, b3, W4, b4 = load_weights(dir)
            test_helper(W1, b1, W2, b2, W3, b3, W4, b4, *v)
        else:
            print("Path does not exist")
# %%
def loss_pred(dir):
    if os.path.exists(dir):
            loss = 0
            W1, b1, W2, b2, W3, b3, W4, b4 = load_weights(dir)
            for y in test_set:
                y = y.reshape(1, 784)
                
                y_pred = forward_helper_full(W1, b1, W2, b2, W3, b3, W4, b4, y, prev = False)
                loss += np.mean(np.square(np.subtract(y, y_pred)))

            print(f'Loss: {loss / len(test_set):.4f}')
    else:
        print("Path does not exist")

# %%
# Latent size of 32 - Loss: 0.0411
# Latent size of 64 - Loss: 0.0283
# %%
