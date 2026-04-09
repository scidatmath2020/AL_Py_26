# -*- coding: utf-8 -*-
"""
Eigenfaces + SVM Lineal - Implementación Didáctica para Álgebra Lineal
Versión con selección automática de r según varianza dada por el usuario
"""

import numpy as np
import os
import cv2
from sklearn import svm
import matplotlib.pyplot as plt

# ====================== CONFIGURACIÓN ======================
RESIZE = 100
TRAIN_DIR = r"......................."
TEST_DIR = r"........................"

celebrities = ["Ben", "Bob", "Simi", "Dipanda", "Richard"]

# ====================== NUEVO PARÁMETRO ======================
TARGET_VARIANCE = 0.99         # ← Cambia este valor (0.80, 0.85, 0.90, etc.)

#%%

def load_images(directory):
    """Carga imágenes, las convierte a grises y las redimensiona a RESIZE×RESIZE"""
    images = []
    filenames = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error al leer: {path}")
                continue
            img = cv2.resize(img, (RESIZE, RESIZE))
            img = img.astype(np.float32).flatten()
            images.append(img)
            filenames.append(fname)
    
    X = np.array(images)
    print(f"Se cargaron {X.shape[0]} imágenes desde: {directory}")
    print(f"Tamaño de cada vector: {X.shape[1]} píxeles\n")
    return X, filenames

#%%

def compute_eigenfaces(X, target_variance=0.85):
    """Calcula eigenfaces seleccionando r automáticamente según la varianza deseada"""
    N, d = X.shape
    print(f"Matriz de datos: {N} imágenes × {d} píxeles")
    
    # Paso 2: Centrado y normalización
    mu = np.mean(X, axis=0)
    A_centered = (X - mu) / 255.0
    
    # Paso 3: Matriz pequeña C'
    C_prime = A_centered @ A_centered.T
    
    # Paso 4: Eigenvalores y eigenvectores
    eigenvalues, eigenvectors = np.linalg.eigh(C_prime)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Selección automática de r según varianza dada por el usuario
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    r = np.argmax(cum_var >= target_variance) + 1
    r = min(r, N-1)
    
    print(f"Variancia objetivo: {target_variance:.1%}")
    print(f"Se usarán {r} componentes principales")
    print(f"Varianza explicada: {cum_var[r-1]:.1%}\n")
    
    # Eigenfaces U_r
    U_r = A_centered.T @ eigenvectors[:, :r]
    
    return U_r, mu, r, eigenvalues

#%%

def project_face(x, mu, U_r):
    """Proyecta una imagen al espacio de eigenfaces"""
    x_centered = (x - mu) / 255.0
    omega = x_centered @ U_r
    return omega

#%%

def train_models(X_train, filenames_train, U_r, mu):
    """Entrena un SVM lineal por cada celebridad (One-vs-Rest)"""
    models = {}
    for celeb in celebrities:
        y = np.array([1 if celeb.lower() in fname.lower() else 0 for fname in filenames_train])
        Omega = np.array([project_face(x, mu, U_r) for x in X_train])
        model = svm.SVC(kernel='linear', class_weight='balanced')
        model.fit(Omega, y)
        models[celeb] = model
    return models

#%%

def recognize_face(test_path, models, mu, U_r):
    """Reconoce una imagen de prueba"""
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    x = cv2.resize(img, (RESIZE, RESIZE)).astype(np.float32).flatten()
    omega_test = project_face(x, mu, U_r)
    
    best_celebrity = None
    best_score = -np.inf
    
    print(f"\nScores para: {os.path.basename(test_path)}")
    for celeb, model in models.items():
        score = model.decision_function([omega_test])[0]
        print(f"   {celeb:10}: {score:6.3f}")
        if score > best_score:
            best_score = score
            best_celebrity = celeb
    
    return best_celebrity, best_score

#%%

def show_mean_face(mu):
    plt.figure(figsize=(5, 5))
    plt.imshow(mu.reshape(RESIZE, RESIZE), cmap='gray')
    plt.title("Cara Promedio (μ)")
    plt.axis('off')
    plt.show()


def show_eigenfaces(U_r, num_faces=None, cols=4):
    """
    Muestra los eigenfaces.
    
    Parámetros:
        U_r : matriz de eigenfaces
        num_faces : número de eigenfaces a mostrar (None = mostrar todos)
        cols : número de columnas en la cuadrícula (por defecto 4)
    """
    if num_faces is None:
        num_faces = U_r.shape[1]  # Mostrar todos los eigenfaces disponibles
    
    num_faces = min(num_faces, U_r.shape[1])  # No exceder los disponibles
    
    # Calcular número de filas necesarias
    rows = (num_faces + cols - 1) // cols
    
    plt.figure(figsize=(cols * 3, rows * 3.5))
    
    for i in range(num_faces):
        eigenface = U_r[:, i].reshape(RESIZE, RESIZE)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    
    plt.suptitle(f"Eigenfaces (Base del espacio reducido) - Mostrando {num_faces} de {U_r.shape[1]}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título superior
    plt.show()
    
#%%

# ====================== EJECUCIÓN PRINCIPAL ======================
print("=== Eigenfaces + SVM Lineal - Selección por varianza ===\n")

# Cargar datos
X_train, filenames_train = load_images(TRAIN_DIR)

# Calcular eigenfaces usando la varianza que indicó el usuario
U_r, mu, r, eigenvalues = compute_eigenfaces(X_train, target_variance=TARGET_VARIANCE)

# Visualizaciones didácticas
show_mean_face(mu)
show_eigenfaces(U_r,num_faces=36)

# Entrenar modelos
models = train_models(X_train, filenames_train, U_r, mu)
print(f"Modelos SVM entrenados correctamente con r = {r}\n")

# Pruebas
print("="*70)
print(f"INICIANDO PRUEBAS (r seleccionado por varianza = {TARGET_VARIANCE:.1%})")
print("="*70)

test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for test_file in test_files:
    test_path = os.path.join(TEST_DIR, test_file)
    result, score = recognize_face(test_path, models, mu, U_r)
    
    if result:
        print(f"→ Decisión final: **{result}** (score = {score:.3f})\n")
    print("-" * 60)