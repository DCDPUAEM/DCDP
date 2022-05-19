# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:39:03 2017

@author: Jorge Hermosillo

Modelo lineales:
    - Modelo basico
    - Perceptron dual
    - Perceptron con kernel
        - kernel lineal
        - kernel gaussiano
        - kernel polinomial
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def datos_lin_separables(clusters=2,samples=100):
    """
    Genera datos linealmente separables
    Se generan hasta 10 clusters distintos c/u con distribución normal N(mu,s)
    Entrada: numero de clusters y numero de muestras por cluster (clase)
    Salida: Lista con matrices de vectores de datos y código de clase por cluster.
    """
    if clusters > 10:
        n = 10
        print('solo se pueden producir hasta 10 clusters')
    elif clusters < 2:
        n = 2
        print('el numero minimo de clusetrs es 2')
    else:
        n = clusters
    
    """    
    arreglo que contiene los valores promedio de cada distribucion de datos
    """
    mus= []   
    
    if n == 2:
        mus.append([-3,-3])
        mus.append([3,3])
    elif n == 3:
        mus.append([-3,-3])
        mus.append([0,3])
        mus.append([5,-3])
    elif n == 4:
        mus.append([-5,-3])
        mus.append([-3,3])
        mus.append([4,-3])
        mus.append([5,4])
    elif n == 5:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
    elif n == 6:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
    elif n == 7:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
    elif n == 8:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
        mus.append([12,6])
    elif n == 9:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
        mus.append([12,6])
        mus.append([-2,8])
    elif n == 10:
        mus.append([-5,-3])
        mus.append([-3,4])
        mus.append([4,-3])
        mus.append([5,4])
        mus.append([1,1])
        mus.append([-9,3])
        mus.append([8,-3])
        mus.append([12,6])
        mus.append([-2,8])
        mus.append([11,0])
        
    mus = np.asarray(mus)
    
    """    
    valor de correlación (rho) entre los datos de una distribución
    """
    rho = np.random.uniform(low=0.2, high=0.8, size=(n,))  
    
    if n==2:
        p=np.random.rand()
        if p < 0.5:
            rho[0]=-rho[0]
            rho[1]=rho[1]
    if n > 3:
        for i in range(n):
            rho[i] *= (-1)**i
    
    """    
    Desviaciones estandar (s) de cada cluster de datos
     - Se genera un vector de dimensión 2xn con valores de s= 1 + un valor aleatorio
     - Luego se transforma el arreglo en una matriz bidimensional con n filas y 2 columnas
       donde n es el número de clusters
    """
    s = np.random.rand(2*n)+1
    s = s.reshape((n,2))
    
    """    
    Matriz de covarianzas
     - Se construye la matriz:
      [[s_0^2   s_0 x s_1 x rho]
       [s_0 x s_1 x rho   s_1^2]]
    """
    cov_list = []
    for i,v in enumerate(s):
        cov = np.zeros((2,2))
        cov[0,0] = v[0]**2
        cov[1,1] = v[1]**2
        cov[0,1] = v[0]*v[1]*rho[i]
        cov[1,0] = cov[0,1]
        cov_list.append(cov)
    
    """    
    Arreglo de salida
    """
    L = []
    
    clases = np.arange(n,dtype='int')
    if n==2:
        clases[0]=-1
        
    """    
    Se genera un arreglo con el valor de clase por el número de muestras
    """
    y = np.repeat(clases[0],samples).reshape(samples,1).astype(int)
    
    for i in range(1,n):
        yp = np.repeat(clases[i],samples).reshape(samples,1).astype(int)
        y = np.hstack((y,yp))
    
    """    
    Se genera un arreglo de clusters, que son distribuciones normales incluyendo el valor de la clase 
    """
    for i in range(n):
        X = np.random.multivariate_normal(mus[i], cov_list[i],samples)
        X = np.hstack((X,y[:,i].reshape(samples,1).astype(int)))
        L.append(X)
        
    return L

def split_train_test(X,y,Tp):
    """
    Obten datos de entrenamiento y prueba
    Entrada: datos centrados, clases (+/-1) y proporcion de entrenamiento (Tp)
    Salida: valores de entrenamiento y prueba
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i in range(len(X)):
        X1 = X[i]
        y1 = y[i]
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=Tp/100.,shuffle=True)
        X_train.append(X1_train)
        X_test.append(X1_test)
        y_train.append(y1_train)
        y_test.append(y1_test)    
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train = X_train.ravel().reshape(-1,2)
    y_train = y_train.ravel()
    X_test = X_test.ravel().reshape(-1,2)
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test
  
def datos_solapados(n=2,samples=100):
    """
    Genera datos solapados (no linealmente separables)
    Entrada: numero de muestras
    Salida: vectores de datos clase 1 y 2.
    """    
    #Gaussiana 1
    mu1=np.array([3.0,5.0])
    s1=np.array([1.0,2.5])
    #matriz de covarianzas centrada en mu
    corr=0.42
    co_v= s1[0]*s1[1]*corr
    cov1 = [[s1[0]**2,co_v],[co_v,s1[1]**2]]
    
    mu2=np.array([4.0,4.0])
    s2=np.array([0.8,1.7])
    #matriz de covarianzas centrada en mu
    corr=0.3
    co_v= s2[0]*s2[1]*corr
    cov2 = [[s2[0]**2,co_v ],[co_v, s2[1]**2]]
    
    cov_list = [cov1,cov2]
    mus = [mu1,mu2]

    L = []
    
    clases = np.arange(2,dtype='int')
    clases[0]=-1
    y = np.repeat(clases[0],samples).reshape(samples,1).astype(int)
    for i in range(1,2):
        yp = np.repeat(clases[i],samples).reshape(samples,1).astype(int)
        y = np.hstack((y,yp))
    
    for i in range(2):
        X = np.random.multivariate_normal(mus[i], cov_list[i],samples)
        X = np.hstack((X,y[:,i].reshape(samples,1).astype(int)))
        L.append(X)
        
    return np.array(L)


def datos_correl(samples=100,corr=0.6,centro=[0.,0.],sigma=[2.5,1.]):
    """
    Genera datos correlacionados
    Entrada: numero de muestras, correlacion entre datos (+/-)
             y centro de los datos (medias)
    Salida: vector de datos clase 1
    """    
    #Gaussiana 1
    mu1=np.array([centro[0],centro[1]])
    s1=np.asarray(sigma)
    print('factor de correlacion en los datos rho = ',corr)
    #matriz de covarianzas centrada en mu
    cov1 = [[s1[0]**2    , s1[0]*s1[1]*corr], 
           [s1[0]*s1[1]*corr  , s1[1]**2]]
    
    X = np.random.multivariate_normal(mu1, cov1,samples)
    return X

def datos_NO_correl(samples=100,centro=[0.,0.],sigma=[1.,1.]):
    """
    Genera datos correlacionados
    Entrada: numero de muestras, centro de los datos (medias)
    Salida: vector de datos clase 1
    """    
    #Gaussiana 1
    mu1=np.asarray(centro)[0]
    mu2=np.asarray(centro)[1]
    s1=np.asarray(sigma)[0]
    s2=np.asarray(sigma)[1]
    #matriz de datos    
    X1 = np.random.normal(mu1, s1,samples)
    Y1 = np.random.normal(mu2, s2,samples)
    X = np.c_[X1,Y1]
    return X


def normaliza_min_max(X):
    """
    Normalizacion min-max
    Entrada: arreglo numpy de datos 2D
    Salida: datos normalizados entre 0 y 1
    """
    if X.ndim != 1:
        for j in range(X.shape[1]):
            xb=X[:,j]
            X[:,j]=(xb-np.amin(xb))/(np.amax(xb)-np.amin(xb))
    else:
        X=X/np.sum(X)
    return X

def inv_(S):
    """
    Inversion de una matriz 2D
    Entrada: Matriz de datos S
    Salida: inversa de S 
    """
    det=S[0,0]*S[1,1]-S[0,1]*S[1,0]
    a=S[0,0]
    b=S[0,1]
    c=S[1,0]
    d=S[1,1]
    S_1=1/det*np.array([[d,-b],[-c,a]])
    return S_1

def scatter(X):
    Xs=np.dot(X.T,X)
    return Xs

def normaliza(X):
    """
    Normalizacion basada en mean y std
    Entrada: arreglo numpy de datos 2D
    Salida: datos centrados y normalizados 
    """
    Xs=scatter(X)
    Xs_1=inv_(Xs)
    X=np.dot(X,Xs_1)
    return X

def rota(x,c,angle):
    """
    Rotacion respecto de un centro
    Entrada: punto (x,y) a rotar, centro y angulo de rotacion
    Salida: coordenadas del punto rotado
    """
    theta = (angle/180.) * np.pi
    #coordenadas homegeneas para incluir rotacion y translacion
    rotM = np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta),  np.cos(theta)]])
    t=np.append(c,[1]).reshape(3,1)
    r=np.append(rotM,[0,0]).reshape(3,2)
    trans=np.hstack((r,t))
    #trasnlacion para centrar punto en origen
    a=x-c
    o=np.append(a,[1]).reshape(3,1)
    o=np.dot(trans,o)
    return o[:2]    

def centra_datos(X):
    """
    Elimina promedios de los datos por feature X[:,j]
    Entrada: matriz de datos X
    Salida: matriz de datos centrada Xc
    """
    if X.ndim != 1:
        Xm=[X[:,j].mean() for j in range(X.shape[1])]
        Xm=np.asarray(Xm)
        Xc=X-Xm
    else:
        Xc=X-X.mean()
    return Xc


if __name__ == "__main__":
        
    #my_path='/Users/jorge/Google Drive/Clases/MACHINE LEARNING/Sesiones/Imagenes/'
    X1,y1,X2,y2=datos_lin_separables(100,0.52)
    #X1,y1,X2,y2=datos_solapados(100)
    #X1,y1,X2,y2=datos_no_lin_separables(100)
    
    X_train,y_train,X_test,y_test=split_train_test(X1,y1,X2,y2,80)        
    print('|X_train|= {} ; |X_test|= {}'.format(len(X_train),len(X_test)))
    
    #X_train=centra_datos(X_train)
    #X_test=centra_datos(X_test)
    
    plt.scatter(X_train[:,0][y_train>0], X_train[:,1][y_train>0],facecolor='orangered', marker='$\\bigoplus$', edgecolor='none',s=90,label='B2')
    plt.scatter(X_train[:,0][y_train<0], X_train[:,1][y_train<0],facecolor='royalblue', marker='$\\ominus$', edgecolor='none',s=90,label='B1')
    

    plt.show()
