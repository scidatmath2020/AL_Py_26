# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:37:21 2026

@author: Usuario
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os

'''
###############################################################################
##########      Leer archivo e indicar columnas que no se usarán      #########
###############################################################################
'''

## Carpeta donde está el archivo
os.chdir(r"............")

## leer el archivo
df = pd.read_csv("............csv")

df.columns
df.head()

#%%
###############################################################################
###############################################################################
###############################################################################

cols_not = [] # columnas a excluir. Dejar vacío si no hay

###############################################################################
###############################################################################
###############################################################################


#%%
cols_use = [x for x in df.columns if x not in cols_not]

X = df[cols_use]

scaler = StandardScaler()
Z = scaler.fit_transform(X)
    
# PCA con todos los componentes posibles
pca_full = PCA()
pca_full.fit(Z)

varianza_acum = np.cumsum(pca_full.explained_variance_ratio_)


'''
###############################################################################
##########     Seleccionar cuánta varianza se quiere explicar      ############
###############################################################################
'''

min_varianza = 0.8 # 80% de la varianza de R

###############################################################################
###############################################################################
###############################################################################


k_varianza = np.argmax(varianza_acum >= min_varianza) + 1

diff = np.diff(varianza_acum)
k_codo = np.argmax(diff < 0.01) + 1 

k_recomendado = min(k_varianza, k_codo)




