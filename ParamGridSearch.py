import random
from AutoEncoderV3 import AutoEncoder, LinearLayer, Sigmoid


def get_r():
    return random.uniform( 0.001, 0.01 )

def get_batch_size():
    return random.choice( [ 16, 32, 64, 128, 256, 512 ] )

def get_n_epochs():
    return random.randint( 1, 10 )

def get_layers():
    n_hidden = random.randrange( 1, 10, 2 )
    layer_sizes = [
        784,
        *[ random.choice( [ 16, 32, 64, 128, 256, 512 ] ) for _ in range( n_hidden ) ],
        784
    ]
    n_layers = n_hidden + 2
    layers = [ LinearLayer( *layer_sizes[ i:i + 2 ] ) for i in range( n_layers - 1 ) ]    # Generate linear layers
    layers = list( zip( layers, [ Sigmoid() ] * n_layers ) )   # Pair a sigmoid layer with each linear layer
    layers = [ x for t in layers for x in t ]    # Flatten list of layers
    center = len( layers ) // 2
    return layers[ center: ], layers[ :center ]    # Split list into encoder and decoder sets

encoder_layers, decoder_layers = get_layers()
print( "Encoder Layers" )
for l in encoder_layers:
    print( l )
print( "Decoder Layers" )
for l in decoder_layers:
    print( l )