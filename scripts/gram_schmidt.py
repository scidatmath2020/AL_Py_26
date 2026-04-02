# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:53:24 2026

@author: Usuario
"""

import numpy as np

def gram_schmidt(beta):
    n = len(beta)
    u1 = beta[0]/np.linalg.norm(beta[0])
    base_ortonormal = [u1]
    for i in range(1,n):
        ui = beta[i]-sum([np.dot(beta[i],u)*u for u in base_ortonormal])
        ui = ui / np.linalg.norm(ui)
        base_ortonormal.append(ui)
    return base_ortonormal
        
beta = [[1,2,3,4],[2,1,0,-1],[1,0,2,1]]

base_ortonormal = gram_schmidt(beta)
base_ortonormal

np.dot(base_ortonormal[0],base_ortonormal[1])
np.dot(base_ortonormal[0],base_ortonormal[2])
np.dot(base_ortonormal[1],base_ortonormal[2])

np.linalg.norm(base_ortonormal[0])
np.linalg.norm(base_ortonormal[1])
np.linalg.norm(base_ortonormal[2])