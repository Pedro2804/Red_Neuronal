import numpy as np
from scipy import signal

def full_conv(in_, w):
    zin = np.shape(in_)
    zw  = np.shape(w)

    #wr  = np.zeros((zw[0], zw[1]))

    #wr  = 0 * w
    #x   = np.zeros((zin(1)+zw(1)-1, zin(2)+zw(2)-1));

    #rotar w
    #for kx in range(zw[0]):
    #    for ky in range(zw[2]):
    #        wr[kx][ky] = w[zw[1]+1-kx, zw[2]+1-ky]

    in_aux = np.zeros((zin[0]+(zw[0]-1)*2, zin[1]+(zw[1]-1)*2))
    in_aux[zw[0]-1:zw[0]+zin[0], zw[1]-1:zw[1]+zin[1]] = in_
    x = signal.convolve2d(in_aux, w, 'valid')
    #x = np.conv2(in_aux,rot90(rot90(w)),'valid');
    return x