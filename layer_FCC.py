import numpy as np

def layer_fcc(in_, out_, gain_):
    X = np.zeros((in_,1))
    Y = np.zeros((out_,1))
    W = gain_ * 2 * np.random.rand(out_, in_) - gain_
    B = gain_ * 2 * np.random.rand(out_, 1) - gain_
    
    return X, Y, W, B