# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:11:19 2026

@author: Usuario
"""

import numpy as np


A = np.array([[2,1],
              [1,2]
              ])

v = np.array([[1],
              [1]])


np.dot(A,v)

#%%

eigenvals, eigenvecs = np.linalg.eig(A)

eigenvals
eigenvecs

np.dot(A,eigenvecs[:,0])
3*eigenvecs[:,0]

np.dot(A,eigenvecs[:,1])
1*eigenvecs[:,1]

#%%

Q = eigenvecs  # para matrices simétricas, np.linalg.eig devuelve Q ortogonal

np.dot(Q,Q.T)

np.linalg.norm(Q[:,0])
np.linalg.norm(Q[:,1])

np.dot(Q[:,0],Q[:,1])

D = np.diag(eigenvals)

np.dot(np.dot(Q,D),Q.T)










