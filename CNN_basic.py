import view_dataset as vds
import layer_CNN as CNN
import layer_FCC as FCC
import numpy as np
import random
import tensorflow as tf

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

if test_set==0:
    if sobre_train==0:
        cnn_D0, cnn_M0 = 1, 10
        cnn_D1, cnn_M1 = cnn_M0, 10
        cnn_D2, cnn_M2 = cnn_M1, 10
        
        X0, Y0, W0, B0, R0    = CNN.layer_CNN(1, cnn_M0, cnn_D0, 28, 9, 0, 1) #ok<*ASGLU>
        X1, Y1, W1, B1, R1    = CNN.layer_CNN(1, cnn_D1, cnn_M1, 20, 5, 2, 1)
        X2, Y2, W2, B2, R2    = CNN.layer_CNN(1, cnn_D2, cnn_M2, 16, 3, 2, 1)
        
        X3, Y3, W3, B3       = FCC.layer_FCC(1960, 100, 1)
        X4, Y4, W4, B4       = FCC.layer_FCC(100, 100, 1)
        X5, Y5, W5, B5       = FCC.layer_FCC(100, 10, 1)

#Storage variables
Z1 = np.zeros(28,28,1)    # Variable used for display the input image in 
                            # the training of the CNN
if test_set==0:
    # Iterations for training the CNN
    iteraciones = 500e3
else:
    # Iterations for test after the training the CNN
    iteraciones = 5e3

E          = np.zeros(iteraciones,1);  # Loss function of the training set
Etest      = np.zeros(iteraciones,1);  # Loss function of the test set 
yCNN       = np.zeros(10,iteraciones);  # Output of the CNN 
yDPN       = np.zeros(10,iteraciones);  # Desired output 
sourc      = np.zeros(iteraciones,2);  # Input image source

#Training of the CNN
for K in range(iteraciones):
    #LR = LR + 0.2000e-07;   # The learning rate LR increases 0.2000e-07 
                            # each iteration
    try_nan = 0;            #Flag to avoid not a number (NaN) values
    
    # Obtain the input image from the training set by random selection
    while try_nan<1:
        #sp  = floor(rand(1,1)*(10e3-1))+1
        sp = int(random.random() * (10**4 - 1)) + 1 #Variente de la funcion floor
        X0 = dataset["Amnist"][:, :, sp]
        # YD = (1+exp(-ZP_lab(:,sp))).^-1   

        yd  = dataset["label_A"][sp]

        # YD = 1./(1+exp(-  (2.*yd_trt-1)   ));
        YD = [0,0,0,0,0,0,0,0,0,0]
        YD[yd] = 1
        #YD = FCC.switch(yd)

        if np.isnan(X0)==0:
            try_nan=1
        else:
            keep1 = sp

    try_nan = 0
    # Obtain the input image from the test set by random selection
    while try_nan < 1:
        #sp_test  = floor(rand(1,1)*(59999-1))+1;
        sp_test = int(random.random() * (59999 - 1)) + 1
        X0_test  = dataset["Bmnist"][:, :, sp_test]
        # YD_test  = (1+exp(-ZP_lab(:,sp_test))).^-1; 

        yd  = dataset["label_B"][sp_test]
        YD_test = [0,0,0,0,0,0,0,0,0,0]
        YD_test[yd] = 1
        if np.isnan(X0_test)==0:
            try_nan=1
        else:
            keep1 = sp_test

    sourc[K][1] = sp
    sourc[K][2] = sp_test

    #######################################################################
    #                         Test run of the CNN
    #
    for km in range(cnn_M0):
        sm1 = 0 * Y0[:, :, 1]
        for kd in range(cnn_D0):
            #-----------------------------------------------#
            am1 = np.zeros((9,9));                          #
            for q1 in range(9):                             #
                for q2 in range(9):                         #
                    am1[q1][q2] = W0[9-q1+1][9-q2+1][kd][km]#
            #-----------------------------------------------#
            sm1 += np.conv2(X0_test[:, :, kd], am1, mode='valid')

        Y0[:, :, km] = np.maximum(sm1 + B0[km],0)
        X1[:, :, km] = Y0[:, :, km]
        # [X1(:,:,km),R1(:,:,:,km)] = max_pool(Y0(:,:,km),2);

    for km in range(cnn_M1):
        sm1 = 0 * Y1[:, :, 1]
        for kd in range(cnn_D1):
            #-----------------------------------------------#
            am1 = np.zeros((5,5));                          #
            for q1 in range(5):                             #
                for q2 in range(5):                         #
                    am1[q1][q2] = W1[5-q1+1][5-q2+1][kd][km]#
            #-----------------------------------------------#
            sm1 +=np.conv2(X1[:, :, kd], am1, mode='valid')
        Y1[:, :, km] = np.maximum(sm1 + B1[km],0)
        X2[:, :, km] = Y1[:, :, km]
        # [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);

    for km in range(cnn_M2):
        sm2 = 0 * Y2[:, :, 1]
        for kd in range(cnn_D2):
            #-----------------------------------------------#
            am2 = np.zeros((3,3))                           #
            for q1 in range(3):                             #
                for q2 in range(3):                         #
                    am2[q1][q2] = W2[3-q1+1][3-q2+1][kd][km]#
            #-----------------------------------------------#
            sm2 += np.conv2(X2[:, :, kd],am2, mode = 'valid')
        Y2[:, :, km] = np.maximum(sm2 + B2[km],0)
    X3 = np.reshape(Y2,(Y2.size, 1))


    # Verifica si una GPU está disponible
    if tf.config.list_physical_devices('GPU'):
        # Mueve las matrices a la GPU
        X3g = tf.constant(X3, dtype=tf.float32, device="/GPU:0")
        W3g = tf.constant(W3, dtype=tf.float32, device="/GPU:0")
    else:
        print("No se detectó una GPU disponible. Las matrices se mantienen en la CPU.")
    #X3g = gpuArray(X3);
    #W3g = gpuArray(W3);
    Y3g = pagefun(@mtimes,W3g,X3g)
    Y3  = gather(Y3g)
    Y3  = np.maximum(Y3 + 1 * B3, 0)
    
    X4  = Y3;
    X4g = gpuArray(X4);
    W4g = gpuArray(W4);
    Y4g = pagefun(@mtimes,W4g,X4g);
    Y4  = gather(Y4g);
    Y4  = max(Y4 + 1.*B4,0);
    
    X5 = Y4;
    X5g = gpuArray(X5);
    W5g = gpuArray(W5);
    Y5g = pagefun(@mtimes,W5g,X5g);
    Y5  = gather(Y5g);
    
    
    # Y5       = (1+exp(c1.*(-Y5-B5))).^-1;
    Y5 = exp(c1.*(Y5+B5))./sum(exp(c1.*(Y5+B5)));
    Etest(K) = 0.5*mean( (YD_test-Y5).^2 );