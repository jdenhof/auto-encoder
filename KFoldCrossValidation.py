import numpy as np
import random
from keras.datasets import mnist
from AutoEncoderV3 import AutoEncoder, LinearLayer, Sigmoid


rng = np.random.default_rng()


# Load MNIST dataset from keras.
def fix( set ):
    return set.reshape( -1, set.shape[ 1 ]**2 )
def load_MNIST():
    ( train, _ ), ( test, _ ) = mnist.load_data()
    return fix( train ), fix( test )
train_set, test_set = load_MNIST()
assert train_set.shape == ( 60000, 784 ), f"{ train_set.shape } != ( 60000, 784 )"
assert test_set.shape == ( 10000, 784 ), f"{ test_set.shape } != ( 10000, 784 )"


model = AutoEncoder(
    epochs=5,
    optimizer="ADAM",
    encode_layers=[
        LinearLayer( 784, 256 ),
        Sigmoid(),
        LinearLayer( 256, 32 ),
        Sigmoid()
    ],
    decode_layers=[
        LinearLayer( 32, 256 ),
        Sigmoid(),
        LinearLayer( 256, 784 ),
        Sigmoid()
    ]
)
def k_fold( k, set ):
    # Randomize samples across set and split into k groups.
    set = np.copy( set )
    rng.shuffle( set )
    folds = np.split( set, k )
    for f in folds[ 1: ]:
        model.train_adam( f, f.shape[ 0 ] / 100 )



k_fold( 5, train_set )