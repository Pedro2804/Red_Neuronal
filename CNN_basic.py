import view_dataset as vds
import layer_CNN as CNN
import layer_FCC as FCC
import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt

dataset = vds.cargar_archivos()

#Variable initialization
LR	= 10e-3;        #Initial learning rate
MT	= 100;          #Loss function modifier for backpropagation algorithm
# c1	= 1.4e-4;       #Constant for adjusting the output layer of the CNN
c1	= 1e-5; 

test_set    = 0;    #Flag to select between training the CNN or only test 
                    #the CNN
sobre_train = 0;    #Flag to train the CNN from the actual kernels and 
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

if test_set == 0 :
    if sobre_train == 0 :
        cnn_D0, cnn_M0 = 1, 10
        cnn_D1, cnn_M1 = cnn_M0, 10
        cnn_D2, cnn_M2 = cnn_M1, 10
        
        X0, Y0, W0, B0 = CNN.layer_CNN(1, cnn_M0, cnn_D0, 28, 9, 1) #ok<*ASGLU>
        X1, Y1, W1, B1 = CNN.layer_CNN(1, cnn_D1, cnn_M1, 20, 5, 1)
        X2, Y2, W2, B2 = CNN.layer_CNN(1, cnn_D2, cnn_M2, 16, 3, 1)
        
        X3, Y3, W3, B3 = FCC.layer_FCC(1960, 100, 1)
        X4, Y4, W4, B4 = FCC.layer_FCC(100, 100, 1)
        X5, Y5, W5, B5 = FCC.layer_FCC(100, 10, 1)

#Storage variables
Z1 = np.zeros((28,28,1))    # Variable used for display the input image in 
                            # the training of the CNN
if test_set == 0 :
    # Iterations for training the CNN
    iteraciones = 500e3
else:
    # Iterations for test after the training the CNN
    iteraciones = 5e3

E          = np.zeros((iteraciones,1));  # Loss function of the training set
Etest      = np.zeros((iteraciones,1));  # Loss function of the test set 
yCNN       = np.zeros((10,iteraciones));  # Output of the CNN 
yDPN       = np.zeros((10,iteraciones));  # Desired output 
sourc      = np.zeros((iteraciones,2));  # Input image source

#Training of the CNN
for K in range(iteraciones):
    #LR = LR + 0.2000e-07;   # The learning rate LR increases 0.2000e-07 
                            # each iteration
    try_nan = 0;            #Flag to avoid not a number (NaN) values
    
    # Obtain the input image from the training set by random selection
    while try_nan < 1 :
        #sp  = floor(rand(1,1)*(10e3-1))+1
        sp = int(random.random() * (10**4 - 1)) + 1 #Variente de la funcion floor
        X0 = dataset["Amnist"][:, :, sp]
        # YD = (1+exp(-ZP_lab(:,sp))).^-1   

        yd  = dataset["label_A"][sp]

        # YD = 1./(1+exp(-  (2.*yd_trt-1)   ));
        YD = [0,0,0,0,0,0,0,0,0,0]
        YD[yd] = 1
        #YD = FCC.switch(yd)

        if np.isnan(X0) == 0 :
            try_nan = 1
        else:
            keep1 = sp

    try_nan = 0
    # Obtain the input image from the test set by random selection
    while try_nan < 1:
        #sp_test  = floor(rand(1,1)*(59999-1))+1;
        sp_test = int(random.random() * (60000 - 1)) + 1
        X0_test  = dataset["Bmnist"][:, :, sp_test]
        # YD_test  = (1+exp(-ZP_lab(:,sp_test))).^-1; 

        yd  = dataset["label_B"][sp_test]

        YD_test = [0,0,0,0,0,0,0,0,0,0]
        YD_test[yd] = 1

        if np.isnan(X0_test) == 0:
            try_nan=1
        else:
            keep1 = sp_test

    sourc[K][1] = sp
    sourc[K][2] = sp_test

    #######################################################################
    #                         Test run of the CNN
    #
    for km in range(cnn_M0):
        sm1 = np.zeros_like(Y0[:, :, 0])
        for kd in range(cnn_D0):
            #-----------------------------------------------#
            am1 = np.zeros((9,9))                           #
            for q1 in range(9):                             #
                for q2 in range(9):                         #
                    am1[q1][q2] = W0[8-q1][8-q2][kd][km]#
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X0_test[:, :, kd], am1, 'valid')

        Y0[:, :, km] = np.maximum(sm1 + B0[km],0)
        X1[:, :, km] = Y0[:, :, km]
        # [X1(:,:,km),R1(:,:,:,km)] = max_pool(Y0(:,:,km),2);

    for km in range(cnn_M1):
        sm1 = np.zeros_like(Y1[:, :, 0])
        for kd in range(cnn_D1):
            #-----------------------------------------------#
            am1 = np.zeros((5,5))                          #
            for q1 in range(5):                             #
                for q2 in range(5):                         #
                    am1[q1][q2] = W1[4-q1][4-q2][kd][km]#
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X1[:, :, kd], am1, 'valid')
        Y1[:, :, km] = np.maximum(sm1 + B1[km],0)
        X2[:, :, km] = Y1[:, :, km]
        # [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);

    for km in range(cnn_M2):
        sm2 = np.zeros_like(Y2[:, :, 0]) 
        for kd in range(cnn_D2):
            #-----------------------------------------------#
            am2 = np.zeros((3,3))                           #
            for q1 in range(3):                             #
                for q2 in range(3):                         #
                    am2[q1][q2] = W2[2-q1][2-q2][kd][km]#
            #-----------------------------------------------#
            sm2 += signal.convolve2d(X2[:, :, kd], am2, 'valid')
        Y2[:, :, km] = np.maximum(sm2 + B2[km], 0)
    X3 = np.reshape(Y2,(Y2.size, 1))

    Y3 = W3 * X3
    #Función RELU: convierte todos los valores negativos en 0, dejando los valores no negativos sin cambios.
    Y3 = np.maximum(Y3 + 1. * B3, 0)
    
    X4  = Y3
    Y4 = W4 * X4
    Y3 = np.maximum(Y4 + 1. * B4, 0)

    X5  = Y4
    Y5 = W5 * X5

    Y5 = np.exp(c1 * (Y5 + B5)) / np.sum(np.exp(c1 * (Y5 + B5)), axis=1, keepdims=True)
    #Error cuadrático medio
    Etest[K] = 0.5*(np.mean((YD_test - Y5)**2))

    ###########################################################
    #                     Test run of the CNN
    for km in range(cnn_M0):
        sm1 = np.zeros_like(Y0[:, :, 0])
        for kd in range(cnn_D0):
            #-----------------------------------------------#
            am1 = np.zeros((9, 9))
            for q1 in range(9):
                for q2 in range(9):
                    am1[q1][q2] = W0[8 - q1][ 8 - q2][kd][km]
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X0[:, :, kd], am1, 'valid')
        Y0[:, :, km] = np.maximum(sm1 + B0[km], 0)
        X1[:, :, km] = Y0[:, :, km]

    for km in range(cnn_M1):
        sm1 = np.zeros_like(Y1[:, :, 0])
        for kd in range(cnn_D1):
            #-----------------------------------------------#
            am1 = np.zeros((5,5));                          #
            for q1 in range(5):                             #
                for q2 in range(5):                         #
                    am1[q1][q2] = W1[4-q1][4-q2][kd][km] #
            #-----------------------------------------------#
            sm1 += signal.convolve2d(X1[:,:,kd],am1, 'valid')
        Y1[:,:,km] = np.maximum(sm1 + B1[km],0)
        X2[:, :, km] = Y1[:, :, km]
        # [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);

    for km in range(cnn_M2):
        sm2 = np.zeros_like(Y2[:, :, 0])
        for kd in range(cnn_D2):
            #-----------------------------------------------#
            am2 = np.zeros((3,3));                          #
            for q1 in range(3):                             #
                for q2 in range(3):                         #
                    am2[q1][q2] = W2[2-q1][2-q2][kd][km]#
            #-----------------------------------------------#
            sm2 += signal.convolve2d(X2[:, :, kd],am2, 'valid')
        Y2[:, :, km] = np.maximum(sm2 + B2[km],0)

    X3 = np.reshape(Y2, (Y2.size, 1))

    Y3 = W3 * X3
    Y3 = np.maximum(Y3 + 1. * B3, 0)

    X4  = Y3
    Y4 = W4 * X4
    Y3 = np.maximum(Y4 + 1. * B4, 0)

    X5  = Y4
    Y5 = W5 * X5

    Y5_past = Y5
    #Funcion Softmax
    Y5 = np.exp(c1 * (Y5 + B5)) / np.sum(np.exp(c1 * (Y5 + B5)), axis=1, keepdims=True)

    YD_neg = YD
    Y5_neg = Y5

    E[K] = 0.5*np.mean((YD-Y5)**2 )
    yCNN[:][K] = Y5
    yDPN[:][K] = YD

    if (K-1 % 1e3) == 999:
        Q1 = E[K-999:K]
        Q2 = Etest[K-999:K]

        # Crear la primera subtrama
        plt.subplot(1, 2, 1)

        # Calcular las medias de Q1 y Q2
        mean_Q1 = np.mean(Q1, axis=0)
        mean_Q2 = np.mean(Q2, axis=0)

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
        mxmp[axf] = 1 # Línea 362

        # Mostrar el gráfico
        plt.show()