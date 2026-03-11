# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:37:28 2026

@author: Usuario
"""

import os
from plotnine import *
import numpy as np
import pandas as pd
import math

os.chdir(r"C:\Users\Usuario\Documents\scidata\26_AL_Li\practicas")

cancer = pd.read_csv("breast_cancer_dataset.csv")
    
cancer.head()
cancer.tail(15)

cancer.columns

cancer["mean radius"]

cancer["mean radius"].mean()
cancer["mean radius"].var()
cancer["mean radius"].std()

cancer["mean radius"].cov(cancer["mean perimeter"])
cancer["mean radius"].corr(cancer["mean perimeter"])

cancer["mean radius"].describe()

cancer["mean perimeter"].cov(cancer["mean radius"])



media_radius = cancer["mean radius"].mean()
media_perimeter = cancer["mean perimeter"].mean()

radius_centrado = cancer["mean radius"] - media_radius 
perimeter_centrado = cancer["mean perimeter"] - media_perimeter

np.dot(radius_centrado,perimeter_centrado) / (np.linalg.norm(radius_centrado)*(np.linalg.norm(perimeter_centrado)))

cancer.shape

np.dot(radius_centrado,perimeter_centrado)/543


#%%

X = np.random.uniform(-1,1,1000)
Y = np.sqrt(1-X**2)

mis_datos = pd.DataFrame({"col1":X,"col2":Y})


(
ggplot(data=mis_datos) +
    geom_point(mapping=aes(x="col1",y="col2"))  
)

mis_datos["col1"].corr(mis_datos["col2"])











