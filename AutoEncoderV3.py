from Layer import Layer



class AutoEncoder:
    def __init__ (
        self, 
        encode_layers: list[ Layer ],
        decode_layers: list[ Layer ]
    ):
        self.encoder = encode_layers
        self.decoder = decode_layers
    
    def forward( self, input_batch ):
        for layer in self.encoder:
            input_batch = layer.forward(input_batch)
        for layer in self.decoder:
            input_batch = layer.forward(input_batch)
        return input_batch
    
    def backward( self, error_batch ):
        for layer in reversed( self.decoder ):
            error_batch = layer.backward( error_batch )
        for layer in reversed( self.encoder ):
            error_batch = layer.backward( error_batch )

    def update_all_layers( self, optimizer: str, learning_rate ):
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
