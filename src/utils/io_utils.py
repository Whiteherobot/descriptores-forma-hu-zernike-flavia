"""
Ruta:           src/utils/io_utils.py
Responsabilidad:
    Utilidades de entrada/salida: cargar imágenes en escala de grises, recorrer
    el dataset UPS-Writing-Skills y crear carpetas de resultados.

    No contiene lógica de visión; sólo lectura ordenada de los datos para que el
    resto de módulos reciba arreglos NumPy limpios.
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import config


def ensure_dir(path: Path) -> Path:
    """Crea el directorio (y sus padres) si no existe y lo devuelve."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_gray(path: Path) -> np.ndarray:
    """Carga una imagen y la devuelve en escala de grises (uint8)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img


def load_ups_dataset(ups_dir: Path = config.UPS_DIR,
                     classes: List[str] = None) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Recorre data/ups_writing_skills/all-images/<clase>/*.PNG y devuelve:
        images : lista de imágenes en gris
        labels : lista de etiquetas enteras (índice de la clase)
        names  : nombres de las clases en orden de etiqueta

    Parámetros:
        ups_dir : carpeta 'all-images'
        classes : lista de subcarpetas a usar (por defecto config.UPS_CLASSES)
    """
    classes = classes or config.UPS_CLASSES
    images: List[np.ndarray] = []
    labels: List[int] = []

    for label, cls in enumerate(classes):
        folder = ups_dir / cls
        if not folder.exists():
            raise FileNotFoundError(f"No existe la carpeta de clase: {folder}")
        # Se aceptan PNG/png/jpg por robustez.
        files = sorted([p for p in folder.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        for f in files:
            images.append(load_gray(f))
            labels.append(label)

    return images, labels, list(classes)
