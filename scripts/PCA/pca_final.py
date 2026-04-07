# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:14:03 2026

@author: SciData
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necesario para proyección 3D
import os
import joblib

#%%
'''
###############################################################################
##########      Leer archivo e indicar columnas que no se usarán      #########
###############################################################################
'''

## Carpeta donde está el archivo
os.chdir(r"...................")

## leer el archivo
df = pd.read_csv(".........csv")

'''############################################################################
############################################################################'''

print(df.columns)
print(df.head())

#%%

'''############################################################################
####################   Columnas indeseables si existen   ######################
############################################################################'''

''' - Escribir cols_not = [] si no hay columnas indeseables
    - Si la tabla tiene variable objetivo (target), siempre se incluye en cols_not
'''

cols_not = []

'''############################################################################
############################################################################'''

cols_use = [x for x in df.columns if x not in cols_not]
X = df[cols_use]

print(X.columns)

#%%

'''############################################################################
###############   Definir columna objetivos si es que la hay   ################
############################################################################'''

'''Hacer objetivo = [] si no hay columna objetivo.'''
objetivo = [] # objetivo = df["nombre de la columna objetivo"]

'''############################################################################
############################################################################'''

#%%

'''############################################################################
###############   Escalado y cálculo de todas las componentes   ###############
############################################################################'''

# Escalado
scaler = StandardScaler()
Z = scaler.fit_transform(X)
   
# PCA completo
pca_full = PCA()
pca_full.fit(Z)
varianza_acum = np.cumsum(pca_full.explained_variance_ratio_)


print(pd.DataFrame({"n_componentes":range(1,Z.shape[1]+1),
                              "varianza_acumulada":varianza_acum}))

#%%
'''############################################################################
##################    Elección de la varianza mínima     ######################
############################################################################'''

# x% de la varianza de R que se quiere explicar
min_varianza = 0.95 

'''############################################################################
############################################################################'''

k_varianza = np.argmax(varianza_acum >= min_varianza) + 1

diff = np.diff(varianza_acum)
k_codo = np.argmax(diff < 0.01) + 1

k_recomendado = min(k_varianza, k_codo)

print(f"k por minima varianza de {min_varianza}: {k_varianza}")
print(f"k por minima criterio de codo: {k_codo}  (varianza explicada: {varianza_acum[k_codo-1]:.4f})")
print(f"k recomendado: {k_recomendado}")

#%%

'''############################################################################
################    Elección del número de componentes    #####################
############################################################################'''

'''Puede ser  n_componentes = k_varianza, k_codo, k_recomendado o manual'''

n_componentes = k_recomendado #k_recomendado

'''############################################################################
############################################################################'''

pca = PCA(n_components=n_componentes)
Yk = pca.fit_transform(Z)
   
pca.explained_variance_ratio_.sum()

print(f"Varianza explicada con {n_componentes} componentes: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Forma de Yk: {Yk.shape}")

#%%
'''############################################################################
#############                  BLOQUE DE GRÁFICAS                   ###########
############################################################################'''

if n_componentes == 1:
    print("\n⚠️ Se usa solo una componente principal. No se puede hacer gráfica.")
    
elif n_componentes == 2:
    print("\n📊 Se están usando exactamente 2 componentes; se muestra la gráfica para ambas.")
    dim = 2
    
    # --- Crear la figura 2D ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(Yk[:, 0], Yk[:, 1],
                         c = objetivo if len(objetivo) > 0 else None,  
                         cmap = "viridis" if len(objetivo) > 0 else None,
                         alpha=0.8,
                         edgecolors='k',
                         s=50)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA - Primeras 2 Componentes\n(Varianza explicada por {n_componentes} componentes: {pca.explained_variance_ratio_.sum():.2%})')
    plt.colorbar(scatter, label='Target (objetivo)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:  # n_componentes >= 3
    print(f"\n📊 Se están usando exactamente {n_componentes} componentes; se puede hacer una gráfica de 2 o 3 componentes.")
    
    while True:
        respuesta = input("¿Quieres graficar en 2D o en 3D? (escribe 2 o 3): ").strip()
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
                             c = objetivo if len(objetivo) > 0 else None,  
                             cmap= "viridis" if len(objetivo) > 0 else None,
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
                             c = objetivo if len(objetivo) > 0 else None,  
                             cmap= "viridis" if len(objetivo) > 0 else None,
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


#%%

'''############################################################################
################          GUARDAR ARTEFACTOS       ############################
############################################################################'''

joblib.dump(scaler, "pca_scaler.joblib")
joblib.dump(pca, "pca_model.joblib")
np.save("pca_cols_use.npy", np.array(cols_use)) # guardamos los nombres de columnas
np.save("Yk.npy", Yk)
Yk_tabla = pd.DataFrame(Yk, columns=[f"PC{i+1}" for i in range(n_componentes)])
Yk_tabla.to_csv("Yk.csv", index=False)

if len(objetivo):
    Yk_tabla["objetivo"] = objetivo
    Yk_tabla.to_csv("Yk_objetivo.csv", index=False)
    print("Sí hay variable objetivo")
else:
    print("No hay variable objetivo")
    
print("\n✓ Artefactos guardados correctamente:")
print(" - pca_scaler.joblib")
print(" - pca_model.joblib")
print(" - pca_cols_use.npy")
print(" - Yk.npy")
print(" - Yk.csv")
if len(objetivo):
    print(" - Yk_objetivo.csv")
    

print(" Ahora puedes cerrar el notebook y volver más tarde.")
