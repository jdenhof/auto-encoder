import numpy as np
from keras.datasets import mnist
from AutoEncoderV3 import AutoEncoder
from Layer import LinearLayer, Sigmoid
from Trainer import Trainer
from LossFunctions import MSE
from argparse import ArgumentParser


rng = np.random.default_rng()


parser = ArgumentParser()
parser.add_argument( "-b", "--n-batch", type=int )
parser.add_argument( "-e", "--n-epochs", type=int )
parser.add_argument( "-k", "--n-folds", type=int )
args = parser.parse_args()
n_batch = args.n_batch or 32
n_epochs = args.n_epochs or 10
n_folds = args.n_folds or 5


# Load MNIST dataset from keras.
def fix( set ):
    return set.reshape( -1, set.shape[ 1 ]**2 )
def load_MNIST():
    ( train, _ ), ( test, _ ) = mnist.load_data()
    return fix( train ), fix( test )
train_set, test_set = load_MNIST()
assert train_set.shape == ( 60000, 784 ), f"{ train_set.shape } != ( 60000, 784 )"
assert test_set.shape == ( 10000, 784 ), f"{ test_set.shape } != ( 10000, 784 )"


print( f"{ n_folds }-Fold Cross Validation" )


# Randomize samples across set and split into k groups.
set = np.copy( train_set )
rng.shuffle( set )
folds = np.split( set, n_folds )
for i, fold in enumerate( folds ):
    test = fold
    train = folds.copy()
    train.remove( fold )
    train_loss = 0.0
    model = AutoEncoder(
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
    trainer = Trainer( model, MSE, "ADAM" )
    for set in train:
        train_loss += trainer.train( set, n_batch )
    average_loss = train_loss / n_folds
    test_loss = trainer.train( test, n_batch )
    print(
f"""
  Fold { i }:
    Loss:
      Average:  { average_loss }
      Test Set: { test_loss }
"""
    )