from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def cargar_archivos():
    Amnist = loadmat('Archivos_buap/Amnist.mat')
    index_A = loadmat('Archivos_buap/index_A.mat')
    #Bmnist_V2 = loadmat('Archivos_buap/Bmnist_V2.mat')
    #index_B_V2= loadmat('Archivos_buap/index_B_V2.mat')
    Bmnist = loadmat('Archivos_buap/Bmnist_V2.mat')
    label_B= loadmat('Archivos_buap/index_B_V2.mat')

    label_A = np.zeros((10**4, 1), dtype=int)

    a = 0
    x = 0
    for i in range(10):
        b = int(a + index_A["index_A"][i][0])
        for j in range(a, b):
            label_A[j][0] = x
        x += 1
        a = b
    return {"Amnist": Amnist, "label_A": label_A, "Bmnist": Bmnist, "label_B": label_B}

'''
a = 0
for i in range(10):
    d = int(a + index_A["index_A"][i][0])
    for j in range(a, d):
        print(label_A[j], end=' ')
    print()
    a = d
'''

#plt.imshow(data["Amnist"][:, :, 980], cmap='gray')  # cmap='gray' para mostrar la imagen en escala de grises
#plt.axis('off')  # Para ocultar los ejes
#plt.show()