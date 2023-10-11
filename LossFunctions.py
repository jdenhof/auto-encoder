import numpy as np



def MSE( actual, expected ):
    return np.mean( np.square( expected - actual ) )