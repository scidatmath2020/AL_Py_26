# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:37:20 2026

@author: Usuario
"""

import numpy as np
import pandas as pd
import scipy as sp

def T(vector):
    coord_1 = vector[0] - vector[2] + 2*vector[3]
    coord_2 = vector[1] + 3*vector[2] - vector[3]
    return [coord_1,coord_2]

vector_uno = [1,1,1,1] 
auxiliar = np.diag(vector_uno)
imagenes = [T(auxiliar[x,:]) for x in range(len(vector_uno))]
Matriz = np.array(imagenes).T 

base_kernel = sp.linalg.null_space(Matriz)

Matriz @ base_kernel

rango = np.linalg.matrix_rank(Matriz)
rango

U, S, Vt = sp.linalg.svd(Matriz)
base_imagen = U[:,:rango]  

#%%

def T(vector):
    coord_1 = vector[0] - vector[1] 
    coord_2 = 0
    return [coord_1,coord_2]

vector_uno = [1,1,1] 
auxiliar = np.diag(vector_uno)
imagenes = [T(auxiliar[x,:]) for x in range(len(vector_uno))]
Matriz = np.array(imagenes).T 

base_kernel = sp.linalg.null_space(Matriz)

Matriz @ base_kernel

rango = np.linalg.matrix_rank(Matriz)
rango

U, S, Vt = sp.linalg.svd(Matriz)
base_imagen = U[:,:rango]  

















