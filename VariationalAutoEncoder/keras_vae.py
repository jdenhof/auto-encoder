import tensorflow as tf
import numpy as np

# Define the encoder
def build_encoder(latent_dim):
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    return tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')

# Define the sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the decoder
def build_decoder(latent_dim):
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(28*28, activation='sigmoid')(x)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

# Build the VAE model
latent_dim = 2
encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
z_mean, z_log_var = encoder(inputs)
z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
outputs = decoder(z)
vae = tf.keras.Model(inputs, outputs, name='vae')

# Define the VAE loss function
def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

# Compile the VAE model
vae.compile(optimizer='adam', loss=lambda inputs, outputs: vae_loss(inputs, outputs, z_mean, z_log_var))

# Load your dataset and preprocess it (e.g., MNIST)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Train the VAE
vae.fit(x_train, x_train, epochs=10, batch_size=32, validation_data=(x_test, x_test))
