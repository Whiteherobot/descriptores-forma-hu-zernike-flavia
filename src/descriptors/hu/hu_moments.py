"""
Ruta:           src/descriptors/hu/hu_moments.py
Responsabilidad:
    Calcular los 7 Momentos Invariantes de Hu de una figura y aplicar la
    transformación logarítmica vista en clase para igualar los órdenes de
    magnitud.

    Fundamento (Unidad 3.1 - "Fundamentos de los Momentos Invariantes de Hu"):
        - Momentos RAW:      M_ij = Σ_x Σ_y  x^i · y^j · I(x,y)
        - Centroide:         (x̄, ȳ) = (M10/M00, M01/M00)
        - Momentos centrales (invariantes a traslación): μ_pq
        - Momentos normalizados (invariantes a escala):  η_ij = μ_ij / μ00^(1+(i+j)/2)
        - Los 7 invariantes I1..I7 (invariantes a rotación) según Hu.

    Implementación:
        Se usa cv2.moments() y cv2.HuMoments(), que calculan EXACTAMENTE las
        ecuaciones anteriores (los mismos μ, η e I1..I7 del PDF). Esto mantiene
        el código corto y verificable, y es lo estándar en OpenCV.

    Transformación log (vista en clase):
        h_i' = -sign(h_i) · log10(|h_i|)
        Mitiga que I1 sea ~1e-3 e I7 ~1e-11 (diferencia de órdenes de magnitud).
"""
import cv2
import numpy as np


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """
    Calcula los 7 momentos de Hu de una máscara binaria y devuelve su versión
    log-transformada (vector de 7 floats).

    Entrada:
        mask : máscara binaria uint8 (figura rellena, 255 = objeto).
    """
    # cv2.moments calcula M_ij, mu_ij y nu_ij (η) del documento de clase.
    m = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()   # 7 invariantes I1..I7

    # Transformación logarítmica (igualar órdenes de magnitud).
    eps = 1e-30
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu_log.astype(np.float64)


# Nombre legible de cada componente, útil para tablas/figuras del reporte.
HU_FEATURE_NAMES = [f"Hu{i}" for i in range(1, 8)]
