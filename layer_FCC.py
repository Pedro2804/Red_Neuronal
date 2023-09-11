import numpy as np

def layer_fcc(in_, out_, gain_):
    X = np.zeros((in_,1))
    Y = np.zeros((out_,1))
    W = gain_ * 2 * np.random.rand(out_, in_) - gain_
    B = gain_ * 2 * np.random.rand(out_, 1) - gain_
    
    return X, Y, W, B

def switch(op):
    if op == 0:
        return [1,0,0,0,0,0,0,0,0,0]
    elif op == 1:
        return [0,1,0,0,0,0,0,0,0,0]
    elif op == 2:
        return [0,0,1,0,0,0,0,0,0,0]
    elif op == 3:
        return [0,0,0,1,0,0,0,0,0,0]
    elif op == 4:
        return [0,0,0,0,1,0,0,0,0,0]
    elif op == 5:
        return [0,0,0,0,0,1,0,0,0,0]
    elif op == 6:
        return [0,0,0,0,0,0,1,0,0,0]
    elif op == 7:
        return [0,0,0,0,0,0,0,1,0,0]
    elif op == 8:
        return [0,0,0,0,0,0,0,0,1,0]
    elif op == 9:
        return [0,0,0,0,0,0,0,0,0,1]