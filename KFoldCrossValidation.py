import numpy as np
from keras.datasets import mnist
from AutoEncoder import AutoEncoder
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

train_set = train_set[ :500 ]

from copy import deepcopy
def kfold( set, trainer: Trainer, k=5 ):
    print( f"{ k }-Fold Cross Validation" )
    set = np.copy( set )
    rng.shuffle( set )
    folds = np.split( set, k )
    loss_hists = []
    test_losses = []
    for i, fold in enumerate( folds ):
        print( f"  Fold { i }" )
        # train_sets = deepcopy( folds )
        train_sets = np.array( folds )
        train_sets = np.delete( train_sets, i, 0 )
        trainer_copy = deepcopy( trainer )
        loss_hist = [ trainer_copy.train( set, n_batch ) for set in train_sets ]
        loss_hists.append( loss_hist )
        test_loss = trainer_copy.train( fold, n_batch )
        test_losses.append( test_loss )
    return loss_hists, test_losses

train_history, test_losses = kfold(
    set = train_set,
    trainer = Trainer( AutoEncoder(
        encoder_layers=[
            LinearLayer( 784, 256 ),
            Sigmoid(),
            LinearLayer( 256, 32 ),
            Sigmoid()
        ],
        decoder_layers=[
            LinearLayer( 32, 256 ),
            Sigmoid(),
            LinearLayer( 256, 784 ),
            Sigmoid()
        ]
    ), MSE, "ADAM", n_epochs=1 ),
    k=5
)

from matplotlib import pyplot as plt
for hist in train_history:
    plt.plot( range( len( hist ) ), hist )
plt.xlabel( "K" )
plt.ylabel( "Loss" )
plt.title( f"{ 5 }-Fold Cross Validation" )
plt.savefig( "KFoldLossGraph.png" )