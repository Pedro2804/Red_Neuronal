from scipy.io import loadmat
import numpy as np

def cargar_archivos():
    Amnist = loadmat('Archivos_buap/Amnist.mat')
    index_A = loadmat('Archivos_buap/index_A.mat')
    #Bmnist_V2 = loadmat('Archivos_buap/Bmnist_V2.mat')
    #index_B_V2= loadmat('Archivos_buap/index_B_V2.mat')
    Bmnist_V2 = loadmat('Archivos_buap/Bmnist_V2.mat')
    index_B_V2 = loadmat('Archivos_buap/index_B_V2.mat')

    label_A = np.zeros(int(10e3), dtype=int)

    a = 0
    x = 0
    for i in range(10):
        b = int(a + index_A["index_A"][i][0])
        label_A[a:b] = x
        x += 1
        a = b
        
    return {"Amnist": Amnist["Amnist"], "label_A": label_A, "Bmnist": Bmnist_V2["Bmnist_V2"], "label_B": index_B_V2["index_B_V2"]}

'''
a = 0
for i in range(10):
    d = int(a + index_A["index_A"][i][0])
    for j in range(a, d):
        print(label_A[j], end=' ')
    print()
    a = d
'''
'''
dataset = cargar_archivos()
i=0
for a in dataset["label_A"]:
    print(f"{i}->{a}")
    i += 1
'''

"""
plt.imshow(data["Amnist"][:, :, 980], cmap='gray')  # cmap='gray' para mostrar la imagen en escala de grises

one = np.ones((10,10))
conv = fc.full_conv(data["Amnist"][:, :, 980], (1/100)*one)
plt.imshow(conv, cmap='gray')

plt.axis('off')  # Para ocultar los ejes
plt.show()
"""
