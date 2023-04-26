import numpy as np

class LeastSquaresClassifier():
    def __init__(self, W:np.ndarray=None):
        '''
        W es la matriz de pesos, la cual puede ser especificada desde un principio, esto
        es opcional.
        '''
        self.W = W

    def encoderT(self, y:np.ndarray):
        K = np.max(y) + 1
        identidad = np.eye(K)
        return identidad[y] 

    def fit(self, X:np.ndarray, y:np.ndarray):
        '''
        Este método calcula la matriz de pesos para la matriz de puntos "aumentada" X
        y el conjunto de etiquetas y.
        '''
        T = self.encoderT(y)
        self.W = np.linalg.inv(X.T @ X) @ X.T @ T
        
    def clasifica(self, X:np.ndarray):
        '''
        Este método predice las etiquetas para el conjunto de puntos X
        '''
        return np.argmax(X@self.W,axis=1)
     
