# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:38:32 2026

@author: Usuario
"""

######### Elegir k (elegir el número de componentes)


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
os.chdir(r"C:\Users\Usuario\Documents\scidata\26_AL_Li\data\peso_peces")

## leer el archivo
df = pd.read_csv("Fish.csv")

df.columns
df.head()

#%%
###############################################################################
###############################################################################
###############################################################################

cols_not = ['Species','Weight'] # columnas a excluir. Dejar vacío si no hay

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
varianza_acum

'''
###############################################################################
##########     Seleccionar cuánta varianza se quiere explicar      ############
###############################################################################
'''

min_varianza = 0.99 # 80% de la varianza de R

###############################################################################
###############################################################################
###############################################################################


k_varianza = np.argmax(varianza_acum >= min_varianza) + 1

diff = np.diff(varianza_acum)
k_codo = np.argmax(diff < 0.01) + 1 

k_recomendado = min(k_varianza, k_codo)


k_recomendado