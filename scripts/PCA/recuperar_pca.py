# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:28:46 2026

@author: Usuario
"""

import numpy as np
import pandas as pd

import os
import joblib

'''
###############################################################################
##########                    Cargar los objetos                      #########
###############################################################################
'''

## Carpeta donde están los resultados del PCA
os.chdir(r".............")

###############################################################################
###############################################################################
###############################################################################

scaler = joblib.load("pca_scaler.joblib")
pca    = joblib.load("pca_model.joblib")
cols_use = np.load("pca_cols_use.npy", allow_pickle=True).tolist()
Yk = np.load("Yk.npy")

print(f"Columnas usadas:\n{cols_use}\n")
print(f"Número de componentes: {pca.n_components_}")

mu = scaler.mean_
sigma = scaler.scale_
loadings = pca.components_.T

coef_df = pd.DataFrame(index=cols_use)

for j in range(pca.n_components_):
    qj = loadings[:, j]
    a_j = qj / sigma                    # coeficientes a_ij
    coef_df[f"PC{j+1}"] = a_j
        
b_values = []
for j in range(pca.n_components_):
    qj = loadings[:, j]
    b_j = -np.sum(qj * mu / sigma)
    b_values.append(b_j)

coef_df.loc["(Intercepto)"] = b_values
coef_df.to_csv("PCA_Coeficientes.csv")

Z_hat = pca.inverse_transform(Yk)           # Yk debe existir en memoria
X_hat = scaler.inverse_transform(Z_hat)
X_hat_df = pd.DataFrame(X_hat, columns=cols_use)

X_hat_df.to_csv("X_reconstruido.csv",index=False)
