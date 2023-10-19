from Layer import Layer
from numpy import ndarray


class AutoEncoder:
    def __init__ (
        self, 
        encoder_layers: list[ Layer ],
        decoder_layers: list[ Layer ]
    ):
        self.encoder = encoder_layers
        self.decoder = decoder_layers
    
    def forward( self, input: ndarray ):
        for layer in self.encoder:
            input = layer.forward( input )
        for layer in self.decoder:
            input = layer.forward( input )
        return input
    
    def backward( self, error: ndarray ):
        for layer in reversed( self.decoder ):
            error = layer.backward( error )
        for layer in reversed( self.encoder ):
            error = layer.backward( error )

    def update_all_layers(
        self,
        optimizer: str,
        learning_rate: float
    ):
        for layer in reversed( self.decoder ):
            layer.update( learning_rate, optimizer )
        for layer in reversed( self.encoder ):
            layer.update( learning_rate, optimizer )

    # def save(self):
    #     x = 0
    #     def dir(x):
    #         return f'autoencoder_{self.learning_rate}_{self.epochs}_{x}_{self.optimizer}.pkl'
    #     while(os.path.exists(dir(x))):
    #         x += 1
    #     with open(dir(x), 'wb') as file:
    #         pickle.dump(self, file)

    # def display(self):
    #     plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    #     plt.plot(self.loss_graph.keys(), self.loss_graph.values(), marker='o', linestyle='-')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Loss Per Epoch')
    #     plt.grid(True)
    #     plt.show()
