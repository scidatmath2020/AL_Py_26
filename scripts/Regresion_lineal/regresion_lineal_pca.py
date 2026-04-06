# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:33:33 2026

@author: SciData
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

###############################################################################
#################### Configuración de ruta ####################################
###############################################################################

######## carpeta donde está la tabla
os.chdir(r"...............")

## Leer el archivo de datos originales
X = pd.read_csv("................csv")


#%%
###############################################################################
#################### Cargar datos del PCA #####################################
###############################################################################

df = pd.read_csv("Yk_objetivo.csv")

#%%

y = df['objetivo'].copy()
Y = df.drop(columns=['objetivo']).copy()

print(f"Matriz Y: {Y.shape[1]} componentes")
print(f"Variable objetivo: {y.name}")

#%%
###############################################################################
#################### Entrenar modelo de regresión #############################
###############################################################################
model_full = LinearRegression(fit_intercept=True)
model_full.fit(Y, y)

y_pred = model_full.predict(Y)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\n=== RESULTADOS DEL MODELO PCR ===")
print(f"R²     : {r2:.6f}")
print(f"RMSE   : {rmse:.2f}")

#%%
###############################################################################
#################### Recuperar coeficientes originales ########################
###############################################################################
pca = joblib.load("pca_model.joblib")
scaler = joblib.load("pca_scaler.joblib")

gamma = model_full.coef_                    # coeficientes en espacio de componentes
loadings = pca.components_                  # matriz Q (loadings)
beta_std = loadings.T @ gamma               # coeficientes en espacio estandarizado
std_dev = scaler.scale_                     # desviaciones estándar originales

beta_original = beta_std / std_dev          # coeficientes en escala original

beta0 = model_full.intercept_ - np.dot(scaler.mean_, beta_original)

# Nombres de las variables
cols_use = np.load("pca_cols_use.npy", allow_pickle=True)
#%%
###############################################################################
#################### Guardar artefactos para predicción #######################
###############################################################################

# 1. Guardar coeficientes beta y intercepto
artifacts = {
    'beta': beta_original,           # coeficientes de las variables originales
    'beta0': beta0,                  # intercepto
    'feature_names': cols_use.tolist(),
    'model_type': 'PCR_to_OLS',
    'n_features': len(cols_use),
    'r2': r2,
    'rmse': rmse
}

joblib.dump(artifacts, "pcr_model_artifacts.joblib")
print("\n✓ Artefactos guardados en: pcr_model_artifacts.joblib")

###############################################################################
#################### Guardar CSV con X original + predicciones ################
###############################################################################

X['objetivo_predicho'] = y_pred
X['residuo'] = y.values - y_pred

X.to_csv("original_predicciones.csv", index=False)

###############################################################################
###############################################################################
###############################################################################

# Mostrar resumen final
print("\n=== RESUMEN FINAL ===")
print(f"Modelo entrenado con {len(cols_use)} variables originales")
print(f"R² en entrenamiento : {r2:.6f}")
print(f"Intercepto β₀       : {beta0:.2f}")
print("Coeficientes:")
for name, coef in zip(cols_use, beta_original):
    print(f"   {name:20} : {coef:10.6f}")

