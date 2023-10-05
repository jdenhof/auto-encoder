import numpy as np
from keras.datasets import mnist
from AutoEncoderV3 import AutoEncoder, LinearLayer, Sigmoid
from argparse import ArgumentParser


rng = np.random.default_rng()


parser = ArgumentParser()
parser.add_argument( "n_folds", "-k", "--n-folds", type=int )
parser.add_argument( "n_epochs", "-e", "--n-epochs", type=int )
parser.add_argument( "n_batch", "-b", "--n-batch", type=int )
args = parser.parse_args()


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
    epochs=args.n_epochs,
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


# Randomize samples across set and split into k groups.
set = np.copy( train_set )
rng.shuffle( set )
folds = np.split( set, args.n_folds )
for f in folds[ 1: ]:
    model.train_adam( f, args.n_batch )
model.train_adam( folds[ 0 ], args.n_batch )