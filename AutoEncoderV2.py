# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import os

from torchvision import datasets, transforms

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
input_size = 28*28
hidden_size = 256
latent_size = 48

learning_rate = 0.1
epochs = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

W1 = np.random.randn(input_size, hidden_size)  # Weights for the hidden layer
b1 = np.zeros((1, hidden_size))  # Biases for the hidden layer
W2 = np.random.randn(hidden_size, latent_size)  # Weights for the output layer
b2 = np.zeros((1, latent_size))
W3 = np.random.randn(latent_size, hidden_size)  # Weights for the hidden layer
b3 = np.zeros((1, hidden_size))  # Biases for the hidden layer
W4 = np.random.randn(hidden_size, input_size)  # Weights for the output layer
b4 = np.zeros((1, input_size))

def forward1(W1, b1, W2, b2, W3, b3, W4, b4, y):
    
    z1 = np.dot(y, W1) + b1 # input_size -> hidden_size
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 # hidden_size -> latent_size
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3 # latent_size -> hidden_size
    a3 = sigmoid(z3)  
    z4 = np.dot(a3, W4) + b4 # hidden_size -> input_size
    y_pred = sigmoid(z4)
    return z1, a1, z2, a2, z3, a3, z4, y_pred

def forward2(y):
    return forward1(W1, b1, W2, b2, W3, b3, W4, b4, y)

epochs_time = 0
epochs_loss = 0

for epoch in range(epochs):

    epoch_start= time.time()
    epoch_loss = 0

    for y in train_set:
        
        y = y.reshape(1, 784)
        
        z1, a1, z2, a2, z3, a3, z4, y_pred = forward2(y)

        epoch_loss += np.square(np.subtract(y, y_pred)).mean()

        dLdyp = y_pred - y

        # backwards
        output_error4 = dLdyp * y_pred * np.subtract(1, y_pred)
        weight_error4 = np.dot(a3.T, output_error4)
        input_error4 = np.dot(output_error4, W4.T)

        W4 -= learning_rate * weight_error4
        b4 -= learning_rate * output_error4

        output_error3 = input_error4 * a3 * (1 - a3)
        weight_error3 = np.dot(a2.T, output_error3)
        input_error3 = np.dot(output_error3, W3.T)

        W3 -= learning_rate * weight_error3
        b3 -= learning_rate * output_error3

        output_error2 = input_error3 * a2 * (1 - a2)
        weight_error2 = np.dot(a1.T, output_error2)
        input_error2 = np.dot(output_error2, W2.T)

        W2 -= learning_rate * weight_error2
        b2 -= learning_rate * output_error2

        output_error1 = input_error2 * a1 * (1 - a1)
        weight_error1 = np.dot(y.T, output_error1)

        W1 -= learning_rate * weight_error1
        b1 -= learning_rate * output_error1

    epoch_total = time.time() - epoch_start
    epochs_time += epoch_total
    avg_time = epochs_time / (epoch + 1)
    est_time = (epochs - (epoch + 1)) * avg_time # seconds total
    hours_in_secs = (60*60)
    min_in_secs = 60

    epochs_loss += epoch_loss
    avg_loss = epochs_loss / (epoch + 1)
    
    hours = int(est_time / hours_in_secs)
    mins = int((est_time - hours_in_secs*hours) / min_in_secs)
    seconds = int((est_time - hours_in_secs* hours - min_in_secs*mins))

    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Avg Loss {avg_loss:.4f} Time: {epoch_total:.2f} Avg Time: {avg_time:.2f} Estimated: {hours}:{mins}:{seconds} ")

# %%

def forward_weights(W1, b1, W2, b2, W3, b3, W4, b4, y):
        _, _, _, _, _, _, _, result = forward_helper_full(W1, b1, W2, b2, W3, b3, W4, b4, y)
        return result

def forward_noweights(sample):
        _, _, _, _, _, _, _, result = forward_helper(sample)
        return result

def forward_helper_full(W1, b1, W2, b2, W3, b3, W4, b4, y):
    z1 = np.dot(y, W1) + b1 # input_size -> hidden_size
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2 # hidden_size -> latent_size
    a2 = sigmoid(z2)
    size = int(np.sqrt(len(a2.T)))

    try:
        preview(a2.reshape(size, size))
    except:
        print("cannot preview latent space")
    
    z3 = np.dot(a2, W3) + b3 # latent_size -> hidden_size
    a3 = sigmoid(z3)  
    z4 = np.dot(a3, W4) + b4 # hidden_size -> input_size
    y_pred = sigmoid(z4)
    return z1, a1, z2, a2, z3, a3, z4, y_pred

def forward_helper(y):
    return forward_helper_full(W1, b1, W2, b2, W3, b3, W4, b4, y)

def test_helper(*v, weights = None):
    
    for val in v:
        sample = test_set[val]
        preview(sample.reshape(28, 28))
        
        if weights == None:
            result = forward_noweights(sample)
        else:
            result = forward_weights(*weights, y)

        preview(result.reshape(28, 28))

def load_weights(dir):
    if not os.path.exists(x):
        return None

    with open(dir, 'rb') as file:
        return pickle.load(file)

def test(*v, dir=None):
    weights = None
    if dir != None:
        if os.path.exists(dir):
            weights = load_weights(dir)
    test_helper(*v, weights = weights)


# %%
x = 0
def dir(x):
    return f'model_weights{x}.pkl'

while(os.path.exists(dir(x))):
    x += 1

with open(dir(x), 'wb') as file:
    pickle.dump((W1, b1, W2, b2, W3, b3, W4, b4), file)

# %%
def fw(y):
    _, _, _, _, _, _, _, result = forward(y)
    return result
# %%
loss = 0
for s in test_set:
    y = y.reshape(1, 784)
    result = fw(s)

    loss += np.square(np.subtract(y, y_pred)).mean()
print(loss)
# %%
