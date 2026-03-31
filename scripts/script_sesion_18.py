# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:04:39 2026

@author: Usuario
"""

import numpy as np

A = np.array([
    [3, 7, 2, 9, 5, 1, 8, 4, 6, 2],
    [5, 8, 1, 4, 7, 3, 9, 2, 5, 7],
    [9, 2, 6, 3, 8, 5, 1, 7, 4, 3],
    [1, 5, 4, 7, 2, 9, 3, 6, 8, 5],
    [7, 3, 8, 1, 6, 4, 2, 9, 3, 1],
    [4, 9, 3, 6, 1, 7, 5, 8, 2, 9],
    [6, 1, 7, 5, 9, 2, 4, 3, 7, 6],
    [2, 4, 9, 8, 3, 6, 7, 1, 9, 4],
    [8, 6, 5, 2, 4, 1, 6, 5, 2, 8],
    [3, 7, 2, 4, 5, 8, 1, 6, 3, 5]
])

A

### Por determinante

np.linalg.det(A)

### Por rango

np.linalg.matrix_rank(A)

### Cálculo de la inversa:
    
A_inv = np.linalg.inv(A)

np.round(np.linalg.inv(A),1)    
    
np.round(np.dot(A,A_inv),1)    






