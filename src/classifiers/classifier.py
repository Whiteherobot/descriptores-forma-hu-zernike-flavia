"""
Ruta:           src/classifiers/classifier.py
Responsabilidad:
    Clasificadores "simples" pedidos por la guía (KNN o SVM) para evaluar cada
    descriptor de la Parte 1.

    - KNN con K = 7 (parámetro de la pista del docente).
    - SVM lineal como alternativa de comparación.
    Ambos van precedidos de un StandardScaler porque los descriptores tienen
    escalas muy distintas (p.ej. Hu log vs magnitudes de Zernike); estandarizar
    evita que una característica domine la distancia Euclídea del KNN.

    Estos son métodos de Machine Learning CLÁSICO (no Deep Learning), acordes a
    las restricciones de la práctica. (Unidad 3 - clasificación de descriptores.)
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..utils import config


def build_classifier(kind: str = "knn"):
    """
    Construye un pipeline (estandarización + clasificador).
        kind : 'knn' (K=7) | 'svm' (lineal).
    """
    if kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=config.KNN_K)
    elif kind == "svm":
        clf = SVC(kernel="linear", C=1.0, random_state=config.RANDOM_SEED)
    else:
        raise ValueError(f"Clasificador desconocido: {kind}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
