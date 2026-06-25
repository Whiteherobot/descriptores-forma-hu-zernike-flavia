"""
Ruta:           src/preprocessing/leaf_segmentation.py
Responsabilidad:
    Segmentación de HOJAS (Parte 2, Flavia Dataset). Réplica EXACTA del flujo
    del núcleo C++ (cpp/src/shape_descriptor.cpp::extractLargestContour) para
    que la referencia en Python y la app móvil obtengan el mismo contorno.

    Flujo (guía Parte 2):
        - Reducción de ruido con filtro Gaussiano.
        - Binarización (Otsu para fotos de hojas; umbral adaptativo opcional para
          el dibujo en vivo del móvil).
        - Cierre morfológico.
        - cv::findContours con RETR_EXTERNAL -> contorno externo más grande.

    A diferencia de la Parte 1 (figuras convexas -> envolvente convexa), las
    hojas NO son convexas, por lo que aquí se usa el contorno DIRECTO.

    Unidad: 3.2.
"""
from typing import Optional

import cv2
import numpy as np


def extract_largest_contour(image: np.ndarray,
                            use_adaptive: bool = False) -> Optional[np.ndarray]:
    """
    Devuelve el contorno externo más grande (N,2) o None.

    image : imagen BGR o gris (hoja oscura sobre fondo claro).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if use_adaptive:
        binary = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
    else:
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return None
    return c.reshape(-1, 2)
