# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:54:30 2026

@author: Usuario
"""

#### PCA general


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necesario para proyección 3D
import os
import joblib


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

###############################################################################
####################   Columnas indeseables si existen   ######################
###############################################################################

# Escribir cols_not = [] si no hay columnas indeseables
# Si la tabla tiene variable objetivo (target), siempre se incluye en cols_not

cols_not = ['Species','Weight']

###############################################################################
###############################################################################
###############################################################################

cols_use = [x for x in df.columns if x not in cols_not]
X = df[cols_use]

###############################################################################
###############   Definir columna objetivos si es que la hay   ################
###############################################################################

# dejar comentado si no existe columna objetivo
objetivo = df['Weight']

###############################################################################
###############################################################################
###############################################################################

# Escalado
scaler = StandardScaler()
Z = scaler.fit_transform(X)
   
# PCA completo
pca_full = PCA()
pca_full.fit(Z)
varianza_acum = np.cumsum(pca_full.explained_variance_ratio_)
varianza_acum

#%%
# ================================
# Selección de componentes
# ================================

###############################################################################
##################    Elección de la varianza mínima     ######################
###############################################################################

# x% de la varianza de R que se quiere explicar
min_varianza = 0.8 

###############################################################################
###############################################################################
###############################################################################

k_varianza = np.argmax(varianza_acum >= min_varianza) + 1

diff = np.diff(varianza_acum)
k_codo = np.argmax(diff < 0.01) + 1

k_recomendado = min(k_varianza, k_codo)

###############################################################################
################    Elección del número de componentes    #####################
###############################################################################

# Puede ser  n_componentes = k_varianza, k_codo, k_recomendado o manual

n_componentes = 3#k_recomendado

###############################################################################
###############################################################################
###############################################################################

pca = PCA(n_components=n_componentes)
Yk = pca.fit_transform(Z)
   
pca.explained_variance_ratio_.sum()
   
print(f"Varianza explicada con {n_componentes} componentes: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Forma de Yk: {Yk.shape}")

###############################################################################
################ BLOQUE DE GRÁFICAS ###########################################
###############################################################################

# --- Preguntar al usuario solo cuando n_componentes >= 3 ---

if n_componentes == 2:
    dim = 2
else:
    while True:
        respuesta = input(f"\n n_componentes = {n_componentes}. ¿Quieres graficar en 2D o en 3D? (escribe 2 o 3): ").strip()
        if respuesta == '2':
            dim = 2
            break
        elif respuesta == '3':
            dim = 3
            break
        else:
            print("⚠️ Por favor, escribe solo '2' o '3'")
# --- Crear la figura según la dimensión elegida ---
fig = plt.figure(figsize=(10, 8))
if dim == 2:
    ax = fig.add_subplot(111)
    scatter = ax.scatter(Yk[:, 0], Yk[:, 1],
###############################################################################
###############################################################################
###############################################################################
                         c=objetivo,  # comentar si no hay columna objetivo
###############################################################################
###############################################################################
###############################################################################
                         cmap='viridis',
                         alpha=0.8,
                         edgecolors='k',
                         s=50)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA - Primeras 2 Componentes\n(Varianza explicada por {n_componentes} componentes: {pca.explained_variance_ratio_.sum():.2%})')
    plt.colorbar(scatter, label='Target (objetivo)')
   
elif dim == 3:
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Yk[:, 0], Yk[:, 1], Yk[:, 2],
###############################################################################
###############################################################################
###############################################################################
                         c=objetivo,  # comentar si no hay columna objetivo
###############################################################################
###############################################################################
###############################################################################
                         cmap='viridis',
                         alpha=0.8,
                         edgecolors='k',
                         s=50)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'PCA - Primeras 3 Componentes\n(Varianza explicada por {n_componentes} componentes: {pca.explained_variance_ratio_.sum():.2%})')
    plt.colorbar(scatter, label='Target (objetivo)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

###############################################################################
################          GUARDAR ARTEFACTOS       ############################
###############################################################################

joblib.dump(scaler, "pca_scaler.joblib")
joblib.dump(pca, "pca_model.joblib")
np.save("pca_cols_use.npy", np.array(cols_use)) # guardamos los nombres de columnas
np.save("Yk.npy", Yk)
Yk_tabla = pd.DataFrame(Yk, columns=[f"PC{i+1}" for i in range(n_componentes)])
Yk_tabla.to_csv("Yk.csv", index=False)

###############################################################################
##########       Comentar si no existe columna objetivo              ##########
###############################################################################

Yk_tabla["objetivo"] = objetivo
Yk_tabla.to_csv("Yk_objetivo.csv",index=False)

###############################################################################
###############################################################################
###############################################################################

print("\n✓ Artefactos guardados correctamente:")
print(" - pca_scaler.joblib")
print(" - pca_model.joblib")
print(" - pca_cols_use.npy")
print(" - Yk.npy")
print(" - Yk.csv")
print(" Ahora puedes cerrar el notebook y volver más tarde.")