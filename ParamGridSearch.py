import random
from AutoEncoderV3 import AutoEncoder, LinearLayer, Sigmoid


def get_r():
    return random.uniform( 0.001, 0.01 )

def get_batch_size():
    return random.choice( [ 16, 32, 64, 128, 256, 512 ] )

def get_n_epochs():
    return random.randint( 1, 10 )

def get_layers():
    n_layers = random.randint( 2, 10 )
    layers = [ random.choice( [ random.choice( [ 16, 32, 64, 128, 256, 512 ] ) ] ) for _ in range( n_layers - 2 ) ]
    return [ 784, *layers, 784 ]