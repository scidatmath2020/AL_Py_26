# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:33:34 2026

@author: Usuario
"""

# Cálculo de coordenadas en Python

import numpy as np

base = [[1,2,1],[0,-1,3],[2,1,-1]]

v = np.array([[5],
              [-1],
              [4]
             ])


A = np.column_stack(base)

np.linalg.solve(A, v)

#######################################################

base= [[1,1],[1,-1]]

v= np.array([[2],
[3]
])

A=np.column_stack(base)

print(A)
print(np.linalg.solve(A,v))