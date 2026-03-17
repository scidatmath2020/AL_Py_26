# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:13:44 2026

@author: Usuario
"""

import numpy as np


# Crear dos matrices pequeñas
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])


# Multiplicación con @ (recomendado)
C = A @ B
C

# Multiplicación con np.dot (equivalente)
C_dot = np.dot(A, B)
C_dot

X = np.array([[-3,2],
              [4,-5],
              [-4,-5]
               ])

X.T

######  X*X^T
np.dot(X,X.T)

######  X^T*X
np.dot(X.T,X)

col = np.array([[-5],
                [20]])

np.dot(X,col)

################################################

### matriz de m=3 filas y n=4 columnas
### v debe 4 filas y 1 columna
### u debe 3 filas y 1 columna

A = np.array([[4,2,1,7],
          [1,2,3,9],
          [np.pi,np.exp(1),np.sqrt(2),1]
          ])

A.T

v = np.array([[-2],
              [3],
              [8],
              [-7]
              ])

A_v = np.dot(A,v)

u = np.array([[1],
              [0],
              [1]
              ])

At_u = np.dot(A.T,u)


def producto_interno(vect1,vect2):
    dimension = len(vect1)
    return sum([vect1[x]*vect2[x] for x in range(dimension)]).item()
    
producto_interno(u,A_v)
producto_interno(At_u,v)




