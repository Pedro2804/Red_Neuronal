import view_dataset as vds
import layer_CNN as CNN
import layer_FCC as FCC
import numpy as np
import random

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
        sp = int(random.random() * (10**3 - 1)) + 1 #Variente de la funcion floor
        X0 = dataset["Amnist"][:, :, sp]
        # YD = (1+exp(-ZP_lab(:,sp))).^-1   

        yd  = dataset["label_A"][sp]

        # YD = 1./(1+exp(-  (2.*yd_trt-1)   ));
        YD = FCC.switch(yd)

        if np.isnan(X0)==0:
            try_nan=1
        else:
            keep1 = sp