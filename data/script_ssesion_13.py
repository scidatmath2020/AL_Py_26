# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:56:40 2026

@author: Usuario
"""

import numpy as np
import pandas as pd
import os

os.chdir(r"C:\Users\Usuario\Documents\scidata\26_AL_Li\practicas")

tabla = pd.read_csv("tabla_01.csv")

STD = np.diag(tabla.std())

STD_inv = np.linalg.inv(STD)
np.dot(tabla, STD_inv)


np.dot(tabla, STD_inv)

tabla_transformada = pd.DataFrame(np.dot(tabla, STD_inv))

tabla_transformada.std()
