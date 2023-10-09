import random
from AutoEncoderV3 import AutoEncoder, LinearLayer, Sigmoid


def rand_learning_rate():
    return random.uniform( 0.001, 0.01 )

def rand_batch_size():
    return random.choice( [ 16, 32, 64, 128, 256, 512 ] )

def rand_n_epochs():
    return random.randint( 1, 10 )

def rand_layer_sizes( n_hidden: int ):
    return [
        784,
        *[ random.choice( [ 16, 32, 64, 128, 256, 512 ] ) for _ in range( n_hidden ) ],
        784
    ]

def rand_layers( n_hidden = None ):
    if n_hidden == None:
        n_hidden = random.randrange( 1, 10, 2 )
    layer_sizes = rand_layer_sizes( n_hidden )
    n_layers = len( layer_sizes ) - 1
    layers = list(
        x for t in    # Flatten list
        [ zip(
            [ LinearLayer( *layer_sizes[ i:i + 2 ] ) for i in range( n_layers ) ],    # Generate linear layers
            [ Sigmoid() ] * n_layers    # Pair a sigmoid layer with each linear layer
        ) ]
        for x in t    # End flatten
    )
    center = len( layers ) // 2
    return layers[ center: ], layers[ :center ]    # Split list into encoder and decoder sets

# encoder_layers, decoder_layers = rand_layers()
# print( "Encoder Layers" )
# for l in encoder_layers:
#     print( l )
# print( "Decoder Layers" )
# for l in decoder_layers:
#     print( l )

