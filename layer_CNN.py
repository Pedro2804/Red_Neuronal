import numpy as np
import random

def layer_cnn(N,M,D,size_in,size_filter,reduction,gain):
    ns = size_in-size_filter+1

    X = np.zeros((size_in, size_in, D, N))
    Y = np.zeros((ns,ns,M,N))
    W = gain * 2 * random.rand(size_filter, size_filter, D, M) - gain * 1
    B = gain * 2 * random.rand(M, 1) - gain * 1

    Reg_pool = None
    if reduction > 0:
        Reg_pool = np.zeros((size_in,size_in,3,D,N))

    return X, Y,W, B,Reg_pool