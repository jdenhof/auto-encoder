import NN
import numpy as np
import matplotlib.pyplot as plt

class Model:

    def __init__(self, encoder: NN.Sequential, decoder: NN.Sequential, epochs = None, reconstruction_alpha = None, learning_rate = None, optimizer = None, loss_function = None):
        """
        encoder : NN.Sequential - encoder layers
        decoder : NN.Sequential - decoder layers
        optimizer : "ADAM" | "SGD" | "MINIBATCH" | "BATCH"
        learning_rate : to be applied to each layer
        reconstruction_alpha : 0 = reconstruction_loss | 1 = kl_div | 0.5 = reconstruction_loss + kl_div
        loss_function : NN.LossFunctions.{lossfunction} ex. NN.LossFunctions.MSE
        """
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.reconstruction_alpha = reconstruction_alpha
        self.loss_fn = loss_function
        self.epochs = epochs
        self.loss_graph = {}

        encoder_latent_size = encoder.summary[-1][1]
        decoder_latent_size = decoder.summary[0][0]
        try:
            assert encoder_latent_size == decoder_latent_size
        except:
            print("Encoder latent size not equal to decoder latent size")
            exit()

        latent_size = encoder_latent_size

        self.variational_layers = NN.Parrallel(
            NN.Sequential(NN.LinearLayer(latent_size, latent_size)),
            NN.Sequential(NN.LinearLayer(latent_size, latent_size))
        )

    def encode(self, input):
        input = self.encoder(input)
        z_mean, z_log_var = self.variational_layers(input)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var
    
    def reparameterize(self, z_mu, z_log_var):
        eps = np.random.normal(0, 1, size=z_mu.shape)
        z = z_mu + eps * np.exp(z_log_var / 2.)
        return z    
    
    def forward(self, input):
        encoded, z_mean, z_log_var = self.encode(input)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

    def train(self, train_set, batch_size = 1):

        if self.loss_fn is None:
            def_lf = NN.LossFunctions.MSE
            print(f"Loss function not specified: Default - {def_lf.__name__}")
            self.loss_fn = def_lf

        if self.epochs is None:
            def_ep = 100
            print(f"Epochs not specified: Default - {def_ep}")
            self.epochs = def_ep

        if self.learning_rate is None:
            def_lr = 0.001
            print(f"Learning rate not specified: Default - {def_lr}")
            self.learning_rate = def_lr

        if self.optimizer is None:
            def_op = 'ADAM'
            print(f"Optimizer not specified: Default - {def_op}")
            self.optimizer = def_op

        if self.reconstruction_alpha is None:
            def_ra = 0.5
            print(f"Reconstruction Alpha not specified: Default - {def_ra}")
            self.reconstruction_alpha = def_ra

        if self.optimizer == 'ADAM' and batch_size == 1:
            batch_size = 32
            print(f"Batch size not specified: Default - {batch_size}")

        epochs_loss = 0
        for epoch in range(self.epochs):
            batches = [train_set[i:i+batch_size] for i in range(0, train_set.shape[0], batch_size)]
            epoch_loss = 0
            for batch in batches:
                encoded, z_mean, z_log_var, decoded = self.forward(batch)

                kl_div = -0.5 * np.sum(1 + z_log_var - z_mean ** 2 - np.exp(z_log_var), axis=0)
                batch_size = kl_div.shape[0]
                kl_div = kl_div.mean()

                loss, gradient = self.loss_fn(batch, decoded)
                epoch_loss += loss

                gradient = self.decoder.backward(gradient)

                gradient = self.variational_layers.backward(gradient)
                
                gradient = self.reconstruction_alpha * kl_div +  (1 - self.reconstruction_alpha) * gradient

                gradient = self.encoder.backward(gradient)

                self.encoder.update(self.learning_rate, self.optimizer)
                self.decoder.update(self.learning_rate, self.optimizer)

            self.loss_graph[epoch] = epoch_loss
            epoch_loss = epoch_loss / len(batches)
            epochs_loss += epoch_loss

            print(f"Epoch: {epoch + 1} Loss: {epoch_loss} Avg Loss/Epoch: {epochs_loss/ (epoch + 1)}")

            if epoch > 0:
                self.display()

    def display(self):
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        plt.plot(self.loss_graph.keys(), self.loss_graph.values(), marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Per Epoch')
        plt.grid(True)
        plt.show()

    encoder_model = NN.Sequential(
        NN.LinearLayer(784, 256),
        NN.Sigmoid(),
        NN.LinearLayer(256, 32),
        NN.Sigmoid()
    )   

    decoder_model = NN.Sequential(
        NN.LinearLayer(32, 256),
        NN.Sigmoid(),
        NN.LinearLayer(256, 784),
        NN.Sigmoid()
    )

class Tester:

    def __init__ ( self, vae: Model, test_set: [] ):
        self.vae = vae
        self.test_set = test_set

    def __call__ (self, *tests):
        for test in tests:
            test = self.test_set[test]
            NN.MNIST.preview(test)
            encoded, z_mean, z_log_var, decoded = self.vae.forward(test)
            NN.MNIST.preview(decoded)




            



                


            
                






