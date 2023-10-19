import numpy as np
from math import log2
from AutoEncoder import AutoEncoder



class Trainer:
    def __init__( self, model: AutoEncoder, loss_function, optimizer: str, learning_rate=0.001, n_epochs=100 ) -> None:
        self.model = model
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.loss_graph = [ None ] * n_epochs

    def train( self, train_set, batch_size=32 ):
        if self.optimizer == "minibatch" or self.optimizer == "ADAM":
            assert batch_size > 1, f"Batch size [ { batch_size } ] must be > 1."
            assert log2( batch_size ).is_integer(), f"Batch size [ { batch_size } ] must be an power of 2."
        total_loss = 0.0
        for epoch in range( self.n_epochs ):
            loss = 0.0
            match self.optimizer:
                case "batch":
                    loss = self.batch( train_set )
                case "SGD":
                    loss = self.sgd( train_set )
                case "minibatch" | "ADAM":
                    loss = self.minibatch( train_set, batch_size )
                case _:
                    raise ValueError( f"Unable to match [ { self.optimizer } ] to an optimizer." )
            self.loss_graph[ epoch ] = loss
            total_loss += loss
        return total_loss


    
    def batch( self, set ):
        set_copy = np.copy( set )
        np.random.shuffle( set_copy )
        epoch_loss = self.propagate( set_copy )
        self.model.update_all_layers( self.optimizer, self.learning_rate )
        return epoch_loss



    def minibatch( self, set, batch_size=32 ):
        set_copy = np.copy( set )
        np.random.shuffle( set_copy )
        epoch_loss = 0.0
        if len( set_copy ) % batch_size:
            set_copy = set_copy[ :-( len( set_copy ) % batch_size ) ]
        batches = np.split( set_copy, len( set_copy ) / batch_size )
        np.random.shuffle( batches )
        for i, batch in enumerate( batches ):
            print( f"Batch { i + 1 }/{ len( batches ) }", end="\r" )
            loss = self.propagate( batch )
            self.model.update_all_layers( self.optimizer, self.learning_rate )
            epoch_loss += loss
        return epoch_loss



    def sgd( self, set ):
        set_copy = np.copy( set )
        np.random.shuffle( set_copy )
        epoch_loss = 0.0
        for sample in set_copy:
            loss = self.propagate( sample )
            self.model.update_all_layers( self.optimizer, self.learning_rate )
            epoch_loss += loss
        return epoch_loss


    
    def propagate( self, sample ):
        output = self.model.forward( sample )
        loss = self.loss_function( output, sample )
        gradient = output - sample
        self.model.backward( gradient )
        return loss