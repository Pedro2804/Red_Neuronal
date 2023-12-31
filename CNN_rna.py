#import os

#cwd = os.getcwd()  # Get the current working directory (cwd)
#files = os.listdir(cwd)  # Get all the files in that directory
#print('q: ',cwd)

import view_dataset as vds
import layer_CNN as CNN
import layer_FCC as FCC
import full_conv as FC
import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt
import math

dataset = vds.cargar_archivos()

depurar = 0 # 1: activar variables para depurar
def full_conv2(inn,w):
    zin = np.shape(inn)
    zw = np.shape(w)
    #print("zin",zin)
    #print("zw",zw)
    #wr = w * 0
    #x = zeros(zin(1) + zw(1)-1, zin(2) + zw(2)-1)
    #rotar w
    #for kx = 1
    #
    #
    #
    #
    in_aux = np.zeros((zin[0] + (zw[0]-1)*2, zin[1] + (zw[1]-1)*2))
    #print(in_aux.shape)
    in_aux[zw[0] - 1:zw[0] - 1 + zin[0], zw[1] - 1:zw[1] - 1 + zin[1]] = inn
    #in_aux[zw[0]: zw[0]-1+zin[0], zw[1]:zw[1]-1+zin[1]] = inn
    x = signal.convolve2d(in_aux, w,'valid')
    return x
#Variable initialization
LR	= 10e-3        #Initial learning rate
MT	= 100          #Loss function modifier for backpropagation algorithm
# c1	= 1.4e-4;       #Constant for adjusting the output layer of the CNN
c1	= 1e-5 

test_set    = 0    #Flag to select between training the CNN or only test 
                    #the CNN
sobre_train = 0    #Flag to train the CNN from the actual kernels and 
                    #synaptic weights instead from random values

#Define the architecture of the CNN

#                     [Input: 28x28x1 size]
#                               |
#                               |
#                               V
#            [Conv layer 1: Kernel 9x9x2x10, bias 10x1, 
#           20x20x10 map size, ReLU activation function]
#                               |
#                               |
#                               V
#            [Conv layer 2: Kernel 5x5x10x10, bias 10x1, 
#            16x16x10 map size, ReLU activation function]
#                               |
#                               |
#                               V
#            [Conv layer 3: Kernel 3x3x10x10, bias 10x1, 
#           14x14x10 map size, ReLU activation function]
#                               |
#                               |
#                               V
#         [Rearrange the map of 14x14x10 to a vector of 1960x1]
#                               |
#                               |
#                               V
#       [Full connected layer 1: synaptic weights 100x1960, 
#                bias 100x1,Relu activation funcion]
#                               |
#                               |
#                               V
#       [Full connected layer 2: synaptic weights 100x100, 
#                bias 100x1,Relu activation funcion]
#                               |
#                               |
#                               V
#       [Full connected layer 3: synaptic weights 10x100, 
#                bias 10x1,sigmoid activation funcion]
#                               |
#                               |
#                               V
#                       [Output: 10x1 size]

const = 3e-4

if test_set == 0 :
    if sobre_train == 0 :
        cnn_D0, cnn_M0 = 1, 10
        cnn_D1, cnn_M1 = cnn_M0, 10
        cnn_D2, cnn_M2 = cnn_M1, 10
        
        X0, Y0, W0, B0 = CNN.layer_cnn(1, cnn_M0, cnn_D0, 28, 9, 1)
        X1, Y1, W1, B1 = CNN.layer_cnn(1, cnn_D1, cnn_M1, 20, 5, 1)
        X2, Y2, W2, B2 = CNN.layer_cnn(1, cnn_D2, cnn_M2, 16, 3, 1)
        
        X3, Y3, W3, B3 = FCC.layer_fcc(1960, 100, 1)
        X4, Y4, W4, B4 = FCC.layer_fcc(100, 100, 1)
        X5, Y5, W5, B5 = FCC.layer_fcc(100, 10, 1)
'''
print(f"X0: {X0.shape}")
print(f"X1: {X1.shape}")
print(f"X2: {X2.shape}")
print(f"X3: {X3.shape}")
print(f"X4: {X4.shape}")
print(f"X5: {X5.shape}")

print(f"Y0: {Y0.shape}")
print(f"Y1: {Y1.shape}")
print(f"Y2: {Y2.shape}")
print(f"Y3: {Y3.shape}")
print(f"Y4: {Y4.shape}")
print(f"Y5: {Y5.shape}")

print(f"Y0: {B0.shape}")
print(f"Y1: {B1.shape}")
print(f"Y2: {B2.shape}")
print(f"Y3: {B3.shape}")
print(f"Y4: {B4.shape}")
print(f"Y5: {B5.shape}")

BIN = 5

pesos = [W0, W1, W2, W3,W4,W5]
titulo_histograma = ['Histograma w0', 'Histograma w1', 'Histograma w2', 'Histograma w3','Histograma w4','Histograma w5']
for i, W in enumerate(pesos):
    fW = W.flatten()
    histW, binsW = np.histogram(fW, BIN)
    plt.figure()
    plt.hist(fW, BIN, edgecolor='k')
    plt.xlabel('valor')
    plt.ylabel('Frecuencia')
    plt.title(titulo_histograma[i])
    plt.legend()

plt.show()
'''

#Storage variables
Z1 = np.zeros((28,28,1))    # Variable used for display the input image in 
                            # the training of the CNN
if test_set == 0 :
    # Iterations for training the CNN
    iteraciones = int(500e3)
else:
    # Iterations for test after the training the CNN
    iteraciones = int(5e3)

E          = np.zeros((iteraciones,1));  # Loss function of the training set
Etest      = np.zeros((iteraciones,1));  # Loss function of the test set 
yCNN       = np.zeros((10,iteraciones));  # Output of the CNN 
yDPN       = np.zeros((10,iteraciones));  # Desired output 
sourc      = np.zeros((iteraciones,2));  # Input image source

contador_mostrar = 0

#iteraciones = 1
#Training of the CNN
for K in range(iteraciones):
    #LR = LR + 0.2000e-07;   # The learning rate LR increases 0.2000e-07 
                            # each iteration
    try_nan = 0;            #Flag to avoid not a number (NaN) values
    
    # Obtain the input image from the training set by random selection
    
    #sp  = floor(rand(1,1)*(10e3-1))+1
    sp = math.floor(random.random() * (int(10e3) - 1)) + 1 #Variente de la funcion floor
    X0 = dataset["Amnist"][:, :, sp]

    yd  = dataset["label_A"][sp]
    

    # YD = (1+exp(-ZP_lab(:,sp))).^-1   

    

    # YD = 1./(1+exp(-  (2.*yd_trt-1)   ));
    YD = np.zeros((10,), dtype=int)
    if depurar == 1:
        yd = 0
    YD[yd] = 1
    YD = np.reshape(YD,(-1,1))

    # Obtain the input image from the test set by random selection
    #sp_test  = floor(rand(1,1)*(59999-1))+1;
    sp_test = math.floor(random.random() * (59999 - 1)) + 1
    X0_test  = dataset["Bmnist"][:, :, sp_test]
    # YD_test  = (1+exp(-ZP_lab(:,sp_test))).^-1; 

    yd  = dataset["label_B"][sp_test]

    YD_test = np.zeros((10,), dtype=int)
    if depurar == 1:
        yd = 0
    YD_test[yd] = 1
    YD_test = np.reshape(YD_test,(-1,1))

    sourc[K, 0] = sp
    sourc[K, 1] = sp_test

    #######################################################################
    #                         Test run of the CNN
    #
    if depurar == 1:
        Xop = np.zeros((28,28))
        #X0 test
        # 
        #           Ciclos for para probar el código
        #
        for i in range(28):
            for j in range(28):
                Xop[i,j] = i + j * 28
        X0_test = Xop
        X0 = Xop

        cfil = 3e-4
        c1 = c1*1e-3
        Wop = np.zeros((9,9,1,10))
        for k1 in range(1):
            for p1 in range(10):
                for i1 in range(9):
                    for j1 in range(9):
                        Wop[i1,j1,k1,p1] = i1 + 8 * j1
        W0 = Wop*cfil

        W1p = np.zeros((5,5,10,10))
        for k1 in range(10):
            for p1 in range(10):
                for i1 in range(5):
                    for j1 in range(5):
                        W1p[i1,j1,k1,p1] = i1 + 4 * j1
        W1 = W1p*cfil

        W2p = np.zeros((3,3,10,10))
        for k1 in range(10):
            for p1 in range(10):
                for i1 in range(3):
                    for j1 in range(3):
                        W2p[i1,j1,k1,p1] = i1 + 2 * j1
        W2 = W2p*cfil

        W3p = np.zeros((100,1960))
        for ki in range (100):
            for kj in range(1960):
                W3p[ki,kj] = ki + kj*10
        W3 = W3p*cfil

        W4p = np.zeros((100,100))
        for ki in range (100):
            for kj in range(100):
                W4p[ki,kj] = ki + kj*10
        W4 = W4p*cfil

        W5p = np.zeros((10,100))
        for ki in range (10):
            for kj in range(100):
                W5p[ki,kj] = ki + kj*10
        W5 = W5p*cfil
        B0 = 0*B0
        B1 = 0*B1
        B2 = 0*B2
        B3 = 0*B3
        B4 = 0*B4
        B5 = 0*B5

        W0 = W0 - 0.0108
        W1 = W1 - 0.003
        W2 = W2 - 9.0000e-04
        W3 = W3 - 2.9534
        W4 = W4 - 0.1633
        W5 = W5 - 0.1498

        # Prueba dos pesos
        W0p = np.zeros((9*9*1*10,1))
        W1p = np.zeros((5*5*10*10,1))
        W2p = np.zeros((3*3*10*10,1))
        W3p = np.zeros((100*1960,1))
        W4p = np.zeros((100*100,1))
        W5p = np.zeros((10*100,1))

        for k1 in range(9*9*1*10):
            W0p[k1] = k1
        for k1 in range(5*5*10*10):
            W1p[k1] = k1
        for k1 in range(3*3*10*10):
            W2p[k1] = k1
        for k1 in range(100*1960):
            W3p[k1] = k1
        for k1 in range(100*100):
            W4p[k1] = k1
        for k1 in range(10*100):
            W5p[k1] = k1

        W0 = np.reshape(W0p,(9,9,1,10),order='F')
        W0 = W0*cfil
        W1 = np.reshape(W1p,(5,5,10,10),order='F')
        W1 = W1*cfil
        W2 = np.reshape(W2p,(3,3,10,10),order='F')
        W2 = W2*cfil
        W3 = np.reshape(W3p,(100,1960),order='F')
        W3 = W3*cfil
        W4 = np.reshape(W4p,(100,100),order='F')
        W4 = W4*cfil
        W5 = np.reshape(W5p,(10,100),order='F')
        W5 = W5*cfil

        W0 = W0 - 0.1213
        W1 = W1 - 0.3748
        W2 = W2 - 0.1348
        W3 = W3 - 29.3998
        W4 = W4 - 1.4998
        W5 = W5 - 0.1498
        #print(W2[0:3,0:3,4,4])
        #break
        #'''
    '''
    X0p = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            X0p[i, j] = i + j * 28
    X0_test = X0p
    X0 = X0p

    #plt.imshow(X0p/812)
    #plt.show()

    W0p = np.zeros((9,9,1,10))
    for k in range(10):
        for i in range(9):
            for j in range(9):
                W0p[i, j, 0, k] = i + j * 8
    W0 = W0p*const

    W1p = np.zeros((5,5,10,10))
    for i in range(5):
        for j in range(5):
            for k in range(10):
                for l in range(10):
                    W1p[i, j, k, l] = i + j * 4
    W1 = W1p*const

    W2p = np.zeros((3,3,10,10))
    for i in range(3):
        for j in range(3):
            for k in range(10):
                for l in range(10):
                    W2p[i, j, k, l] = i + j * 2
    W2 = W2p*const

    W3p = np.zeros((100, 1960))
    for i in range(100):
        for j in range(1960):
            W3p[i, j] = i + j * 10
    
    W4p = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            W4p[i, j] = i + j * 10
    
    W5p = np.zeros((10, 100))
    for i in range(10):
        for j in range(100):
            W5p[i, j] = i + j * 10
    
    W3 = W3p*const
    W4 = W4p*const
    W5 = W5p*const

    B0 = 0*B0
    B1 = 0*B1
    B2 = 0*B2
    B3 = 0*B3
    B4 = 0*B4
    B5 = 0*B5
    '''
    # Primera CNN para validar
    for km in range(cnn_M0):
        sm1 = np.zeros((20,20))
        for kd in range(cnn_D0):
            #-----------------------------------------------#
            am1 = np.zeros((9,9))                           #
            for q1 in range(9):                             #
                for q2 in range(9):                         #
                    am1[q1, q2] = W0[8-q1, 8-q2, kd, km]#
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X0_test, am1, 'valid')

        Y0[:, :, km, 0] = np.maximum(sm1 + B0[km,0],0)
        X1[:, :, km, 0] = Y0[:, :, km, 0]
        # [X1(:,:,km),R1(:,:,:,km)] = max_pool(Y0(:,:,km),2);

    #print(X1[0:3,0:3,5,0])
    #break
    #print(X1[0:3, 0:3, 0, 0])
    #print(am1[0:3, 0:3])
    #break

    for km in range(cnn_M1):
        sm1 = np.zeros_like(Y1[:, :, 0, 0])
        for kd in range(cnn_D1):
            #-----------------------------------------------#
            am1 = np.zeros((5,5))                           #
            for q1 in range(5):                             #
                for q2 in range(5):                         #
                    am1[q1, q2] = W1[4-q1, 4-q2, kd, km]#
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X1[:, :, kd, 0], am1, 'valid')

        Y1[:, :, km, 0] = np.maximum(sm1 + B1[km, 0],0)
        X2[:, :, km, 0] = Y1[:, :, km, 0]
        # [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);

    #print(X2[0:3,0:3,5,0])
    #break
    for km in range(cnn_M2):
        sm2 = np.zeros_like(Y2[:, :, 0, 0]) 
        for kd in range(cnn_D2):
            #-----------------------------------------------#
            am2 = np.zeros((3,3))                           #
            for q1 in range(3):                             #
                for q2 in range(3):                         #
                    am2[q1, q2] = W2[2-q1, 2-q2, kd, km]#
            #-----------------------------------------------#
            sm2 += signal.convolve2d(X2[:, :, kd, 0], am2, 'valid')

        Y2[:, :, km, 0] = np.maximum(sm2 + B2[km, 0], 0)
    #print(Y2[0:3,0:3,5,0])
    #break
    #print(Y2[:, :, 0, 0])

    #print(Y2[0:3, 0:3, 0, 0])
    #break

    #for i in range(10):
    #    np.savetxt(f'capa3_{i}.txt', Y2[:, :, i, 0])

    X3 = np.reshape(Y2.T,(Y2.size, 1))

    Y3 = W3 @ X3
    #print(f'Y3: {Y3[0:10]}')
    #Función RELU: convierte todos los valores negativos en 0, dejando los valores no negativos sin cambios.
    Y3 = np.maximum(Y3 + 1. * B3, 0)
    #print(f'Y3 max: {Y3[0:10]}')
    #print(Y3[0:3,0])
    #break

    
    X4  = Y3
    Y4 = W4 @ X4
    Y4 = np.maximum(Y4 + 1. * B4, 0)
    #print(Y4[97:100,0])
    #break

    #print(W5[:3, :3])
    
    #print(X3[:3])
    #print(Y2[:3, :3, 0, 0])

    X5  = Y4
    Y5 = W5 @ X5


    Y5 = np.exp(c1 * (Y5 + B5)) / np.sum(np.exp(c1 * (Y5 + B5)))
    #Error cuadrático medio
    Etest[K] = 0.5*(np.mean((YD_test - Y5)**2))
    #print(Y5)
    #print(Etest[K])
    #break

    ###########################################################
    # Segunda CNN para entrenar
    for km in range(cnn_M0):
        sm1 = np.zeros((20,20))
        for kd in range(cnn_D0):
            #-----------------------------------------------#
            am1 = np.zeros((9,9))                           #
            for q1 in range(9):                             #
                for q2 in range(9):                         #
                    am1[q1, q2] = W0[8-q1, 8-q2, kd, km]#
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X0, am1, 'valid')

        Y0[:, :, km, 0] = np.maximum(sm1 + B0[km,0],0)
        X1[:, :, km, 0] = Y0[:, :, km, 0]
        # [X1(:,:,km),R1(:,:,:,km)] = max_pool(Y0(:,:,km),2);

    #print(X1[0:3,0:3,5,0])
    #break
    #print(X1[0:3, 0:3, 0, 0])
    #print(am1[0:3, 0:3])
    #break

    for km in range(cnn_M1):
        sm1 = np.zeros_like(Y1[:, :, 0, 0])
        for kd in range(cnn_D1):
            #-----------------------------------------------#
            am1 = np.zeros((5,5))                           #
            for q1 in range(5):                             #
                for q2 in range(5):                         #
                    am1[q1, q2] = W1[4-q1, 4-q2, kd, km]#
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X1[:, :, kd, 0], am1, 'valid')

        Y1[:, :, km, 0] = np.maximum(sm1 + B1[km, 0],0)
        X2[:, :, km, 0] = Y1[:, :, km, 0]
        # [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);

    #print(X2[0:3,0:3,5,0])
    #break
    for km in range(cnn_M2):
        sm2 = np.zeros_like(Y2[:, :, 0, 0]) 
        for kd in range(cnn_D2):
            #-----------------------------------------------#
            am2 = np.zeros((3,3))                           #
            for q1 in range(3):                             #
                for q2 in range(3):                         #
                    am2[q1, q2] = W2[2-q1, 2-q2, kd, km]#
            #-----------------------------------------------#
            sm2 += signal.convolve2d(X2[:, :, kd, 0], am2, 'valid')

        Y2[:, :, km, 0] = np.maximum(sm2 + B2[km, 0], 0)
    #print(Y2[0:3,0:3,5,0])
    #break
    #print(Y2[:, :, 0, 0])

    #print(Y2[0:3, 0:3, 0, 0])
    #break

    #for i in range(10):
    #    np.savetxt(f'capa3_{i}.txt', Y2[:, :, i, 0])

    X3 = np.reshape(Y2.T,(Y2.size, 1))

    Y3 = W3 @ X3
    #print(f'Y3: {Y3[0:10]}')
    #Función RELU: convierte todos los valores negativos en 0, dejando los valores no negativos sin cambios.
    Y3 = np.maximum(Y3 + 1. * B3, 0)
    #print(f'Y3 max: {Y3[0:10]}')
    #print(Y3[0:3,0])
    #break

    
    X4  = Y3
    Y4 = W4 @ X4
    Y4 = np.maximum(Y4 + 1. * B4, 0)
    #print(Y4[97:100,0])
    #break

    #print(W5[:3, :3])
    
    #print(X3[:3])
    #print(Y2[:3, :3, 0, 0])

    X5  = Y4
    Y5 = W5 @ X5


    Y5 = np.exp(c1 * (Y5 + B5)) / np.sum(np.exp(c1 * (Y5 + B5)))
    #Error cuadrático medio
    E[K] = 0.5*(np.mean((YD - Y5)**2))
    #print(Y5)
    #print(E[K])
    #break
    if (K - 1) % 1000 == 999:
        Q1 = E[K - 999:K]
        Q2 = Etest[K - 999:K]
        print('Error: ', np.mean(Q1), np.mean(Q2), K)
    '''
    print(contador_mostrar)
    if (contador_mostrar == 999):
        #if (math.cos(K * math.pi * 0.01) == 0):
        Q1 = E[K-999:K]
        Q2 = Etest[K-999:K]

        # Crear la primera subtrama
        plt.subplot(1, 2, 1)

        # Calcular las medias de Q1 y Q2
        mean_Q1 = np.mean(Q1, axis=0)
        mean_Q2 = np.mean(Q2, axis=0)
        print(E[K])
        print(Etest[K])
        # Crear el gráfico semilogarítmico con el valor de K en el eje x
        K = 123  # Reemplaza 123 con el valor entero de K que desees
        plt.semilogy([K], mean_Q1, 'b.', label='Q1')
        plt.semilogy([K], mean_Q2, 'r.', label='Q2')

        # Etiquetas de los ejes y leyenda
        plt.xlabel('K')
        plt.ylabel('Mean')
        plt.legend()

        axf = np.argmax(Y5_neg) - 1

        mxmp = np.zeros(10)
        mxmp[axf] = 1

        """
        ---------FALTA IMPLEMENTAR-----------
        [YD_neg,mxmp,abs(YD_neg-mxmp)]%#ok
        LR                          % Display the dots of the loss 
                                    % function, the desired value and 
                                    % CNN value of the iteration and 
                                    % the learning rate
        
        hold on
        subplot(1,2,2)
        imshow(X0.*0.5);
        pause(1e-20);
        """

        # Mostrar el gráfico
        plt.show()

        contador_mostrar = 0
    else:
        contador_mostrar += 1
    '''
    
    # Back propagation error
    if test_set == 0:
        YD = np.reshape(YD, (10,1))
        dE5 = (Y5 - YD) * MT
        #print(f'dE5 {dE5[:3]}')
        dF5 = c1 * Y5 * (1 - Y5)

        dC5 = dE5 * dF5
        dW5 = -LR * X5.T * dC5
        dB5 = -LR * dC5

        dC5 = np.reshape(dC5, (10, 1))
        #print(dC5)
        #print(dW5[8:10,98:100])
        #print(dB5)
        #break

        #print(f'W5 shape: {W5.shape}')
        #print(f'dC5 shape: {dC5.shape}')
        #print(f'dC5 {dC5[:3]}')
        #print(f'W5.T {W5.T[:3, :3]}')

        dE4 = W5.T @ dC5

        #print(f'dE4 {dE4[:3]}')

        dF4 = np.sign(Y4)
        dC4 = dE4 * dF4
        #print(dC4[97:100])
        #break

        dW4 = dC4 @ X4.T
        dW4 = -LR * dW4
        dB4 = -LR * dC4
        #print(dW4[97:100,97:100])
        #print(dB4[97:100])
        #break

        #print(f'dW4 {dW4[:3, :3]}')
        #print(f'dB4 {dB4[:3]}')


        #break

        dE3 = W4.T @ dC4
        dF3 = np.sign(Y3)
        dC3 = dE3 * dF3
        #print(dC3[97:100])
        #break

        dW3 = dC3 @ X3.T
        dW3 = -LR * dW3
        dB3 = -LR * dC3
        #print(dW3[97:100,1957:1960])
        #print(dB3[97:100])
        #break

        #dE2f = W3.T @ dC3

        #dE2 = np.reshape(dE2f, (14, 14, 10, 1))
        dE2f = np.transpose(W3) @ dC3 #dC3 @ np.transpose(W3)
        dE2 = np.reshape(dE2f, (14, 14, 10),order='F')
        dF2 = np.sign(Y2)
        #print(dE2[0:3,0:3,0])
        #break
        #print(dE2.shape)
        #print(dF2.shape)
        dC2 = dE2 * dF2[:,:,:,0]
        #print(dC2[0:3,0:3,5])
        #break

        dW2 = np.zeros_like(W2)
        dB2 = np.zeros_like(B2)

        #for km in range(cnn_M2):
        #    dCs2 = np.zeros((14, 14), dtype=int)
        #    
        #    for q1 in range(14):
        #        for q2 in range(14):
        #            dCs2[q1,q2] = dC2[13 - q1][13 - q2][km] #--------------------AQUI-------------------------------------------
        #
        #    for kd in range(cnn_D2):
        #        dW2[:,:,kd,km] = -LR * signal.convolve2d(X2[:,:,kd,0], dCs2, 'valid')
        #    
        #    dB2[km] = -LR * np.sum(dCs2)
        for km in range(cnn_M2):
            dCs2 = np.zeros((14, 14))
            for q1 in range (14):
                for q2 in range (14):
                    dCs2[q1,q2] = dC2[13-q1,13-q2,km]
            
            for kd in range(cnn_D2):
                dW2[:,:,kd,km]=-LR*signal.convolve2d(X2[:,:,kd,0],dCs2, 'valid')
            dB2[km]=-LR*np.sum(sum(dCs2))
        #print(dW2[0:3,0:3,9,9])
        #print(dB2[7:10,0])
        #break
        dE1p = np.zeros_like(X2)
        #print('dE1P:  ',dE1p.shape)
        for kd in range(cnn_D2):
            aq1 = np.zeros_like(dE1p[:, :, 0,0])
            #print('aq1:  ',aq1.shape)
            for km in range(cnn_M2):
                aq1 = aq1 + full_conv2(dC2[:, :, km], W2[:, :, kd, km])
                #aq1 = aq1 + np.correlate(dC2[:, :, km], W2[:, :, kd, km], mode='valid')
            dE1p[:, :, kd,0] = aq1

        dE1 = dE1p
        dF1 = np.sign(Y1)
        dC1 = dE1 * dF1
        #print(dC1[0:3,0:3,5,0])
        #break

        dW1 = np.zeros_like(W1)
        dB1 = np.zeros_like(B1)

        for km in range(cnn_M1):
            dCs1 = np.zeros((16, 16))
            
            for q1 in range(16):
                for q2 in range(16):
                    dCs1[q1, q2] = dC1[15 - q1, 15 - q2, km]

            for kd in range(cnn_D1):
                dW1[:, :, kd, km] = -LR * signal.convolve2d(X1[:, :, kd, 0], dCs1, 'valid')
            
            dB1[km] = -LR * np.sum(dCs1)
          
        #print(dW1[0:3,0:3,9,9])
        #print(dB1[7:10,0])
        #break    
        #dE0p = np.zeros_like(X1)
        #for kd in range(cnn_D1):
        #    aq0 = np.zeros_like(dE0p[:, :, 0, 0])
        #    
        #    for km in range(cnn_M1):
        #        aq0 += FC.full_conv(dC1[:, :, km, 0], W1[:, :, kd, km])
        #    
        #    dE0p[:, :, kd, 0] = aq0

        #dE0 = dE0p
        #dF0 = np.sign(Y0)
        #dC0 = dE0 * dF0
        dE0p = np.zeros_like(X1)
        #print('dE1P:  ',dE1p.shape)
        for kd in range(cnn_D1):
            aq1 = np.zeros_like(dE0p[:, :, 0,0])
            #print('aq1:  ',aq1.shape)
            #print('dc1:  ',dC1.shape)
            for km in range(cnn_M1):
                aq1 = aq1 + full_conv2(dC1[:, :, km,0], W1[:, :, kd, km])
            dE0p[:, :, kd,0] = aq1

        dE0 = dE0p
        dF0 = np.sign(Y0)
        dC0 = dE0 * dF0
        #print(dC0[0:3,0:3,5,0])
        #break

        dW0 = np.zeros_like(W0)
        dB0 = np.zeros_like(B0)

        for km in range(cnn_M0):
            dCs0 = np.zeros((20, 20))
            
            for q1 in range(20):
                for q2 in range(20):
                    dCs0[q1, q2] = dC0[19 - q1, 19 - q2, km]

            for kd in range(cnn_D0):
                dW0[:, :, kd, km] = -LR * signal.convolve2d(X0, dCs0, 'valid')
            
            dB0[km] = -LR * np.sum(dCs0)
        #print(dW0[0:3,0:3,0,9])
        #print(dB0[7:10,0])
        #break

        # if isnan's eliminados

        W5 = W5 + dW5
        B5 = B5 + dB5

        W4 = W4 + dW4
        B4 = B4 + dB4

        W3 = W3 + dW3
        B3 = B3 + dB3

        W2 = W2 + dW2
        B2 = B2 + dB2

        W1 = W1 + dW1
        B1 = B1 + dB1

        W0 = W0 + dW0
        B0 = B0 + dB0