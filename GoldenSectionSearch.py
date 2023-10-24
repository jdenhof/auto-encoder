import numpy as np
from keras.datasets import mnist
from AutoEncoder import AutoEncoder
from Layer import LinearLayer, Sigmoid
from Trainer import Trainer
from KFoldCrossValidation import kfold
from LossFunctions import MSE



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



param_grid = {
    "batch-size": [ 16, 32, 64, 128, 256, 512 ],
    "layer-size": [ 16, 32, 64, 128, 256, 512 ]
}



PHI = ( np.sqrt( 5 ) - 1 ) / 2
def golden_section_search( set, param_grid, k=5, threshold=0.001 ):
    lo_bound = param_grid[ "layer-size" ][  0 ]
    hi_bound = param_grid[ "layer-size" ][ -1 ]
    step = ( hi_bound - lo_bound ) * PHI
    lo_bound_step = lo_bound + step
    hi_bound_step = hi_bound - step
    diff = np.inf
    while diff > threshold:
        model_1 = AutoEncoder(
            encoder_layers=[
                LinearLayer( 784, int( lo_bound ) ),
                Sigmoid()
            ],
            decoder_layers=[
                LinearLayer( int( lo_bound ), 784 ),
                Sigmoid()
            ]
        )
        model_2 = AutoEncoder(
            encoder_layers=[
                LinearLayer( 784, int( hi_bound ) ),
                Sigmoid()
            ],
            decoder_layers=[
                LinearLayer( int( hi_bound ), 784 ),
                Sigmoid()
            ]
        )
        trainer_1 = Trainer( model_1, MSE, "ADAM", n_epochs=1 )
        trainer_2 = Trainer( model_2, MSE, "ADAM", n_epochs=1 )
        _, test_losses_1 = kfold( set, trainer_1, k=k )
        _, test_losses_2 = kfold( set, trainer_2, k=k )
        test_loss_avg_1 = np.average( test_losses_1 )
        test_loss_avg_2 = np.average( test_losses_2 )
        diff = np.absolute( hi_bound_step - lo_bound_step )
        if ( test_loss_avg_1 < test_loss_avg_2 ):
            if ( lo_bound_step < hi_bound_step ):
                hi_bound = hi_bound_step
                step = ( hi_bound - lo_bound ) * PHI
                hi_bound_step = hi_bound - step
            else:
                lo_bound = hi_bound_step
                step = ( hi_bound - lo_bound ) * PHI
                hi_bound_step = lo_bound - step
        else:
            if ( lo_bound_step < hi_bound_step ):
                lo_bound = lo_bound_step
                step = ( hi_bound - lo_bound ) * PHI
                lo_bound_step = lo_bound + step
            else:
                hi_bound = lo_bound_step
                step = ( hi_bound - lo_bound ) * PHI
                lo_bound_step = hi_bound - step


golden_section_search( train_set, param_grid, k=1 )