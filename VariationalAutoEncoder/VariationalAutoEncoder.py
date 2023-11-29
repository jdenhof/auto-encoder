import numpy as np
import matplotlib.pyplot as plt
import random 
import sys 
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

try:
    import NN
except:
    print("Failed to import module NN")

class LossGraph(dict):

    class Colors:
        
        BLUE = 'blue'
        GREEN = 'green'
        RED = 'red'
        CYAN = 'cyan'
        MAGENTA = 'magenta'
        YELLOW = 'yellow'
        BLACK = 'black'
        WHITE = 'white'

        COLORS = [BLUE, GREEN, RED, CYAN, MAGENTA, YELLOW, BLACK]
        
        def random(original = None):

            def get():
                return random.choice(LossGraph.Colors.COLORS)

            color = get()
            while (color == original):
                color = get()
            return color


    def __init__(self, *args):
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.color = None
        self.alt_color = None
        self.unweighted = {}
        
        if len(args) == 3:
            self.initialize_with_all_args(f'{args[0]} Per {args[1]}', args[1], args[0], *args[2:])
        elif len(args) == 4:
            self.initialize_with_all_args(args[0], args[1], args[2], args[3]) 
        else:
            return Exception("Invalid argument length")

    def initialize_with_all_args(self, title, xlabel, ylabel, color = None):
        if color is None:
            color = LossGraph.Colors.random(self.color)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color = color
        self.alt_color = LossGraph.Colors.random(self.color)

    def __str__(self):
        return f"Title: {self.title}, XLabel: {self.xlabel}, YLabel: {self.ylabel}, Color: {self.color}"

    def __setitem__(self, __key, __value):
        if __key == 0 and __value == 0: 
            return
        return super().__setitem__(__key, __value)
    
    def display(self):
        
        for i, plot in enumerate([self, self.unweighted]):
            if i == 1 and len(self.unweighted.keys()) == 0:
                return
            
            if not i == 0:
                color = LossGraph.Colors.random(color)
                title = self.title + " unweighted"
            else:
                color = self.color
                title = self.title

            plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
            plt.plot(plot.keys(), plot.values(), marker='o', linestyle='-', color=color)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.title(title)
            plt.grid(True)
            plt.show()

class BlackBox(NN.Sequential):

    def get_highest_parent_type(obj):
        obj_type = type(obj)
        
        # Continue looping until you reach the highest parent type (base class)
        while obj_type.__base__ is not object:
            obj_type = obj_type.__base__
        
        return obj_type

    def __init__(self, *layers: NN.Layer, name = "BlackBox"):
        for layer in layers:
            try:
                parent_type = BlackBox.get_highest_parent_type(layer)
                assert parent_type == NN.Layer
            except Exception as e:
                raise Exception(f"Layer of type {type(layer)} - parent {parent_type} not allowed in {name}\n{e}")
        super().__init__(*layers)
        self.shape = self._shape()

class Encoder(BlackBox):
    def __init__ (self, *layers: NN.Layer):
        if len(layers) == 0:
            layers = (
                NN.LinearLayer(784, 256),
                NN.Sigmoid(),
                NN.LinearLayer(256, 64)
            )

        super().__init__(*layers, name="Encoder")
        self.latent_size = self.shape[-1][-1]

class Decoder(BlackBox):
    def __init__ (self, *layers: NN.Layer):
        if len(layers) == 0:
            layers = (
                NN.LinearLayer(32, 256),
                NN.Sigmoid(),
                NN.LinearLayer(256, 784),
                NN.Sigmoid()
            )

        super().__init__(*layers, name="Decoder")
        self.latent_size = self.shape[0][0]

class VaritionalLayers(NN.Parrallel):
    def __init__(self, *layers: NN.Layer):
        if len(layers) == 0:
            layers= (
                NN.Sequential(
                    NN.LinearLayer(64, 32)
                ),
                NN.Sequential(
                    NN.LinearLayer(64, 32)
                )
            )

        super().__init__(*layers)
        self.latent_size = self.shape[0][0][1]


class Model:

    def __init__(self, encoder: Encoder, decoder: Decoder, epochs = None, reconstruction_alpha = None, learning_rate = None, optimizer = None, loss_function = None):
        """
        encoder : NN.Sequential - encoder layers
        decoder : NN.Sequential - decoder layers
        optimizer : "ADAM" | "SGD" | "MINIBATCH" | "BATCH"
        learning_rate : to be applied to each layer
        reconstruction_alpha : 0 = reconstruction_loss | 1 = kl_div | 0.5 = reconstruction_loss + kl_div
        loss_function : NN.LossFunctions.{lossfunction} ex. NN.LossFunctions.MSE
        """

        # Model.__init__errors(encoder, decoder, epochs, reconstruction_alpha, learning_rate, optimizer, loss_function, hidden_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.reconstruction_alpha = reconstruction_alpha
        self.loss_fn = loss_function
        self.epochs = epochs
        self.loss_graph = LossGraph('Reconstruction Loss', 'Iteration', 'b')
        self.kl_graph = LossGraph('KL Loss', 'Iteration', 'r')
        self.total_graph = LossGraph('Loss', 'Epoch', 'g')

        # try:
        #     assert self.encoder.latent_size // 2 == self.decoder.latent_size
        # except:
        #     print("Encoder latent size not equal to decoder latent size")
        #     quit()

        self.variational_layers = VaritionalLayers()

        self.model_summary = {
            "Encoder": self.encoder.shape,
            "Decoder": self.decoder.shape,
        }

    def summary(self):
         print(f"Encoder:     {self.encoder.shape}\nDecoder:     {self.decoder.shape}")

    def get_reconstruction_alpha(step):
        beta = 6.5
        k = .0001

        def sig(t, a):
            return 1 / (1 + np.exp(-t*a)) - 0.5
        # Models a sigmoid function that gradually increases
        value = ((0.5) / sig(1, beta)) * (sig(k * step - 1, beta)) + 0.5
        
        if (value <= 0):
            return 0
        
        return value
        
    def encode(self, input):
        output = self.encoder(input)
        mu, log_var = self.variational_layers(output)
        encoded = self.reparameterize(mu, log_var)
        return encoded, mu, log_var
    
    def reparameterize(self, mu, log_var):
        eps = np.random.standard_normal(size=mu.shape)
        # instead of learning the standerd deviation to save computation we learning the log(v^2)
        # we can get standard deviation by e^(log(v^2)/2) = |x|
        z = mu + eps * np.exp(log_var * .5)

        return z    
    
    def __call__(self, input):
        encoded, mu, log_var = self.encode(input)
        decoded = self.decoder(encoded)
        return encoded, mu, log_var, decoded

    def train(self, train_set, batch_size = 1, preview = False):

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

        annealing = False
        if self.reconstruction_alpha is None:
            annealing = True
            print(f"Reconstruction Alpha not specified: Using Default Annealing")

        if self.optimizer == 'ADAM' and batch_size == None:
            batch_size = 32
            print(f"Batch size not specified: Default - {batch_size}")
        
        if batch_size is None:
            batch_size = 1
            print(f"Batch size not specified: Default - {batch_size}")

        input_size = len(train_set[0])
        epochs_loss = 0
        train_set = train_set[:(len(train_set) // batch_size) * batch_size] # Split correctly 
        step = 0
        for epoch in range(self.epochs):
            np.random.shuffle(train_set)
            batches = np.split(train_set, len(train_set) // batch_size)
            epoch_loss = 0
            iterations = 0
            for batch in batches:
                iterations += 1
                Model.assert_test("Batch Shape", (32, 784), batch.shape)
                
                encoded, mu, log_var, decoded = self(batch) # log_var = log(σ^2)

                kl_div = -0.5 * (1 + log_var - np.square(mu) - np.exp(log_var)) # e^log_var = σ^2
                kl_div = np.mean(np.sum(kl_div, axis=0))
                
                reconstruction_loss, gradient = self.loss_fn(batch, decoded)

                Model.assert_test("Gradient from loss fn", gradient.shape, (batch_size, input_size))

                self.kl_graph.unweighted[step] = kl_div
                self.loss_graph.unweighted[step] = reconstruction_loss
                
                Model.assert_test("Gradient to decoder", gradient.shape, batch.shape)

                gradient = self.decoder.backward(gradient)
                
                Model.assert_test("Gradient after decoder", gradient.shape, (batch_size, self.decoder.latent_size))
                
                if annealing:
                    reconstruction_alpha = Model.get_reconstruction_alpha(step)
                else:
                    reconstruction_alpha = self.reconstruction_alpha

                kl_loss = reconstruction_alpha * kl_div
                reconstruction_loss = (1 - reconstruction_alpha) * reconstruction_loss
                
                gradient = kl_loss + (1 - reconstruction_alpha) * gradient
                
                epoch_loss += kl_loss + reconstruction_loss

                self.kl_graph[step] = kl_loss
                self.loss_graph[step] = reconstruction_loss

                Model.assert_test("Gradient to VL", gradient.shape, (batch_size, self.variational_layers.latent_size))

                mu_grad, log_var_grad = self.variational_layers.backward(gradient)
                gradient = mu_grad + log_var_grad

                Model.assert_test("Gradient to Encoder", gradient.shape, (batch_size, self.encoder.latent_size))

                self.encoder.backward(gradient)

                self.variational_layers.update(self.learning_rate, self.optimizer)
                self.encoder.update(self.learning_rate, self.optimizer)
                self.decoder.update(self.learning_rate, self.optimizer)
                step += 1

            Model.assert_test("Number of iterations", iterations, len(train_set) // batch_size)

            epoch_loss = epoch_loss / len(batches)
            self.total_graph[epoch] = epoch_loss
            epochs_loss += epoch_loss

            print(f"Epoch: {epoch + 1} Loss: {epoch_loss} Avg Loss/Epoch: {epochs_loss/ (epoch + 1)} reconstruction alpha: {Model.get_reconstruction_alpha(step - 1)}")
            if preview:
                Model.display(self.total_graph, self.loss_graph, self.kl_graph)

    @staticmethod
    def assert_test(title, predicted, truth) -> bool:
        try:
            assert truth == predicted
        except:
            print(f"{title} -> {predicted} != {truth}")
            raise Exception(f"{title} -> {predicted} != {truth}")
        
        return True
    
    def preview_annealing(self, *range_params):
        graph = LossGraph('Reconstruction Alpha', 'step', 'b')
        for step in range(*range_params):
            graph[step] = Model.get_reconstruction_alpha(step)

        Model.display(graph)

    def display(*loss_graphs: LossGraph):
        for loss_graph in loss_graphs:
            loss_graph.display()

    def __str__(self):
        string = str(self.encoder)
        string += str(self.decoder)
        return string


class Trainer:

    def __init__( self, dataDir = '../data'):
        test_set, train_set = NN.MNIST.get_data(dataDir)
        self.test_set = test_set
        self.train_set = train_set

class Tester:

    def __init__ ( self, vae: Model, test_set: [] ):
        self.vae = vae
        self.test_set = test_set

    def __call__ (self, *tests):
        for test in tests:
            test = self.test_set[test]
            NN.MNIST.preview(test)
            encoded, mu, log_var, decoded = self.vae(test)
            NN.MNIST.preview(decoded)




            



                


            
                






