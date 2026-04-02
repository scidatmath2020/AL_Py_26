# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:16:26 2026

@author: Usuario
"""

import numpy as np

def analizar_sistema(A, b):
    """Analiza el sistema Ax = b. A y b deben ser np.array"""
   
    # Aseguramos que b sea columna
    b_col = b.reshape(-1, 1)
   
    # Matriz aumentada
    Ab = np.hstack((A, b_col))
   
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(Ab)
    n = A.shape[1]
   
    print("=== Análisis del sistema Ax = b ===")
    print(f"Rango de A      : {rank_A}")
    print(f"Rango de [A|b]  : {rank_Ab}")
    print(f"Número de variables : {n}\n")
   
    if rank_A < rank_Ab:
        print("→ El sistema NO tiene solución (inconsistente)")
        return None
   
    elif rank_A == n:
        print("→ El sistema tiene solución ÚNICA")
        x = np.linalg.lstsq(A, b.ravel(), rcond=None)[0]
        print("Solución x =", np.round(x, decimals=8))
        return x
    
    else:
        print("→ El sistema tiene INFINITAS soluciones")
        # Solución particular
        x_part = np.linalg.lstsq(A, b.ravel(), rcond=None)[0]
        print("Una solución particular x =", np.round(x_part, decimals=8))
        
        # Dimensión del kernel
        dim_kernel = n - rank_A
        print(f"   (Hay {dim_kernel} variable(s) libre(s))")
        
        # Base para el kernel (espacio nulo)
        if dim_kernel > 0:
            print("\nBase para el kernel (espacio nulo de A):")
            # Usamos SVD para obtener una base del kernel
            _, _, Vt = np.linalg.svd(A, full_matrices=True)
            kernel_basis = Vt[-dim_kernel:]   # últimas filas de Vt
            for i in range(dim_kernel):
                print(f"v{i+1} =", np.round(kernel_basis[i], decimals=8))
        
        return x_part,kernel_basis


#%%


A = np.array([[1,2],
              [2,-1],
              [3,1]
              ])

b = np.array([5,0,5])

analizar_sistema(A, b)

#%%

A = np.array([[1,1],
              [2,2]
              ])

b = np.array([2,4])

analizar_sistema(A,b)

#%%

A = np.array([[1,0,0,0,0,1,1,1,1,1],
              [0,1,0,0,0,1,2,3,4,5],
              [0,0,1,0,0,1,1,2,3,4],
              [0,0,0,1,0,1,2,1,2,3],
              [0,0,0,0,1,1,1,1,1,1]              
              ])

b = np.array([1,2,3,4,5])

analizar_sistema(A,b)

#%%

A = np.array([[1,1],
              [1,1]
              ])

b = np.array([2,3])

analizar_sistema(A,b)

np.linalg.solve(A,b)





