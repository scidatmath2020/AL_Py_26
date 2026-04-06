# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:01:33 2026

@author: Usuario
"""

import numpy as np
import joblib
import pandas as pd

import os

###############################################################################
#################### Configuración de ruta ####################################
###############################################################################

######## carpeta donde está la tabla con objetivo desconocido
os.chdir(r"...................")

# Leer csv de datos desconocidos
nuevos_datos = pd.read_csv('...............csv')  

###############################################################################
###############################################################################
###############################################################################

#%%

# Cargar los artefactos del modelo
artifacts = joblib.load("pcr_model_artifacts.joblib")

beta = np.array(artifacts['beta'])
beta0 = artifacts['beta0']
feature_names = artifacts['feature_names']

print("Modelo cargado correctamente")
print(f"Variables esperadas: {feature_names}")
print(f"Intercepto: {beta0:.2f}\n")

#%%

###############################################################################
#################### FUNCIÓN DE PREDICCIÓN ####################################
###############################################################################

def predecir_desde_dataframe(df_nuevo):
    """
    Predice para múltiples renglones a la vez (DataFrame).
    El DataFrame debe tener las columnas Variables esperadas
    """
    # Asegurar el orden correcto de columnas
    X = df_nuevo[feature_names].values
    predicciones = beta0 + X @ beta
    return predicciones

#%%
###############################################################################
#################### EJEMPLOS DE USO ##########################################
###############################################################################

preds = predecir_desde_dataframe(nuevos_datos)
nuevos_datos['Prediccion'] = preds
print(nuevos_datos.round(2))

nuevos_datos.to_csv("nuevos_predicciones.csv",index=False)