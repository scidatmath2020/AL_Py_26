# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:10:13 2026

@author: Usuario
"""

import numpy as np


M = np.array([[1,-1],
              [1,1]
              ])


M_inv = np.linalg.inv(M)

M @ M_inv 


#%%%

import numpy as np
import pandas as pd
import scipy as sp

def T(vector):
    coord_1 = vector[4] 
    coord_2 = vector[0] + vector[4]
    coord_3 = vector[1] + vector[4]
    coord_4 = vector[2] + vector[4]
    coord_5 = vector[3] + vector[4]
    return [coord_1,coord_2,coord_3,coord_4,coord_5]

vector_uno = [1,1,1,1,1] 
auxiliar = np.diag(vector_uno)
imagenes = [T(auxiliar[x,:]) for x in range(len(vector_uno))]
Matriz = np.array(imagenes).T 

base_kernel = sp.linalg.null_space(Matriz)

dim_kernel = base_kernel.shape[1]
dim_kernel

Matriz_inv = np.linalg.inv(Matriz)
Matriz_inv

Matriz @ Matriz_inv

u = np.array([[203],
                   [3],
                   [-4],
                   [15],
                   [2026]])

Tu = Matriz @ u
Tu
Matriz_inv @ Tu 











