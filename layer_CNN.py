import numpy as np

"""
Descripción:
  Función que regresa las estructuras a utilizar en CNN
  X: Matriz de la imagen de entrada
  Y: Matriz resultante de aplicar el filtro (operación de convolución) a la imagen de entrada.
  W: Matríz filtro, inicializada con valores aleatorios.

Parámetros:
  N:
  M: No. de conjutnos de filtros
  D: No. de imágenes
  size_in: Tamaño de la imagen de entrada
  size_filter: Tamaño del filtro
  gain:
"""
def layer_cnn(N,M,D,size_in,size_filter,gain):
    ns = size_in-size_filter+1

    X = np.zeros((size_in, size_in, D, N))
    Y = np.zeros((ns,ns,M,N))
    W = gain * 2 * np.random.rand(size_filter, size_filter, D, M) - gain * 1
    B = gain * 2 * np.random.rand(M, 1) - gain * 1

    return X, Y, W, B
