"""
Ruta:           src/preprocessing/segmentation.py
Responsabilidad:
    Convertir una imagen de una figura DIBUJADA A MANO (tinta oscura sobre papel
    claro, trazos que muchas veces NO cierran del todo) en:

        - una SILUETA RELLENA de la figura  -> para Hu y Zernike (momentos de área)
        - el CONTORNO EXTERNO de esa silueta -> para Shape Signature

    Reto principal y decisión de diseño:
        Los dibujos son CONTORNOS ABIERTOS y discontinuos (el bolígrafo deja
        huecos). Rellenar directamente el contorno del trazo sólo rellena la
        "cinta" del trazo, y un flood-fill se "fuga" por los huecos. Como las
        TRES clases de la Parte 1 (círculo, cuadrado, triángulo) son figuras
        CONVEXAS, la silueta se aproxima con la ENVOLVENTE CONVEXA (convex hull)
        del trazo significativo. Esto es robusto a trazos abiertos/rotos y
        preserva la forma (la envolvente convexa de un triángulo es un triángulo,
        la de un cuadrado un cuadrado, etc.).
        (Nota: este criterio es específico de la Parte 1. La Parte 2 trabaja con
         hojas, que NO son convexas, y allí se usa el contorno directo.)

        Pipeline:
            1. Umbral fijo T=130 (pista del docente) -> máscara del trazo.
            2. Descartar manchas pequeñas (motas de ruido) por área.
            3. Envolvente convexa de todo el trazo restante -> silueta SÓLIDA.
            4. Centrado en el CENTROIDE y escalado por el radio máximo
               (invarianza a traslación/escala; encuadre consistente en el disco
               unitario de Zernike).

    Recomendaciones de la guía aplicadas: filtro de mediana contra Salt & Pepper
    antes de umbralizar.

    Unidad: 3.1 (segmentación previa al cálculo de momentos).
"""
from typing import Optional

import cv2
import numpy as np

from ..utils import config


def to_binary(gray: np.ndarray,
              threshold: int = config.T_THRESHOLD,
              median_ksize: int = config.MEDIAN_KSIZE) -> np.ndarray:
    """
    Segmenta la tinta con umbral fijo. Devuelve máscara binaria uint8 {0,255}
    donde 255 = trazo (tinta oscura, gris < T).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if median_ksize and median_ksize >= 3:
        gray = cv2.medianBlur(gray, median_ksize)          # atenúa Salt & Pepper
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def _significant_ink(binary: np.ndarray, min_area_frac: float = 0.05) -> Optional[np.ndarray]:
    """
    Conserva las componentes conexas cuya área sea >= min_area_frac de la mayor
    componente (une los fragmentos del trazo y descarta motas de ruido).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    a_max = int(areas.max())
    keep = np.zeros_like(binary)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= min_area_frac * a_max:
            keep[labels == j] = 255
    return keep


def silhouette(gray: np.ndarray, threshold: int = config.T_THRESHOLD) -> Optional[np.ndarray]:
    """
    Silueta SÓLIDA (envolvente convexa del trazo) sin normalizar tamaño.
    Devuelve máscara uint8 {0,255} o None si no se detecta figura.
    """
    binary = to_binary(gray, threshold)
    # Pequeño cierre para unir trazos casi contiguos antes de filtrar.
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker, iterations=1)

    ink = _significant_ink(binary)
    if ink is None:
        return None
    pts = cv2.findNonZero(ink)
    if pts is None or len(pts) < 3:
        return None

    hull = cv2.convexHull(pts)
    out = np.zeros_like(binary)
    cv2.fillConvexPoly(out, hull, 255)
    return out


def largest_contour(binary: np.ndarray) -> Optional[np.ndarray]:
    """Contorno externo de mayor área (N,2) o None. RETR_EXTERNAL."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def filled_shape_mask(gray: np.ndarray,
                      size: int = config.NORMALIZE_SIZE,
                      threshold: int = config.T_THRESHOLD) -> Optional[np.ndarray]:
    """
    Silueta rellena, centrada en el centroide y escalada al disco unitario.
    Devuelve máscara cuadrada (size x size) uint8 {0,255}, o None.
    """
    sil = silhouette(gray, threshold)
    if sil is None:
        return None
    cnt = largest_contour(sil)
    if cnt is None or cv2.contourArea(cnt) < 1.0:
        return None

    m = cv2.moments(sil, binaryImage=True)
    if m["m00"] <= 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

    d = cnt.astype(np.float64) - np.array([cx, cy])
    max_r = float(np.sqrt((d ** 2).sum(axis=1)).max())
    if max_r <= 0:
        return None

    target_r = (size / 2.0) * 0.95
    scale = target_r / max_r
    M = np.array([[scale, 0.0, size / 2.0 - scale * cx],
                  [0.0, scale, size / 2.0 - scale * cy]], dtype=np.float64)
    return cv2.warpAffine(sil, M, (size, size), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def preprocessed_contour(gray: np.ndarray,
                         threshold: int = config.T_THRESHOLD) -> Optional[np.ndarray]:
    """
    Contorno externo (N,2) de la silueta sólida, para Shape Signature. None si no
    hay figura. Usar la silueta (en vez del trazo crudo) da un contorno limpio y
    cerrado.
    """
    sil = silhouette(gray, threshold)
    if sil is None:
        return None
    return largest_contour(sil)
