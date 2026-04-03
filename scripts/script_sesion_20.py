# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:54:45 2026

@author: Usuario
"""

import numpy as np

A_4menor = np.array([[4,2,2,4],
                     [2,1,1,2],
                     [2,1,1,2],
                     [4,2,2,4]
                     ])

np.linalg.det(A_4menor)

def calculo_menores(matriz):
    n = matriz.shape[0]
    return [np.linalg.det(matriz[0:i+1,0:i+1]) for i in range(n)]

A = np.array([[4,2,2,4],
              [2,1,1,2],
              [2,1,1,2],
              [4,2,2,4]
                     ])

calculo_menores(A)

A = np.array([[2,1],
              [1,2]
                     ])

calculo_menores(A)


#%%

def proyeccion_ortogonal(vector_base,u):
    auxiliar = np.dot(vector_base,u) / np.linalg.norm(vector_base)**2
    return auxiliar * vector_base

vector_base = np.array([1,2])

u = np.array([5.72,3.38])

proyeccion_ortogonal(vector_base, u)

#%%

import os 
import pandas as pd

os.chdir(r"C:\Users\Usuario\Documents\scidata\26_AL_Li\practicas")

cancer = pd.read_csv("breast_cancer_dataset.csv")

mi_cancer = cancer[["mean radius","mean texture", "mean perimeter"]]

S = np.cov(mi_cancer,rowvar=False)

calculo_menores(S)







