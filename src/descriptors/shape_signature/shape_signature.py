"""
Ruta:           src/descriptors/shape_signature/shape_signature.py
Responsabilidad:
    Descriptor "Shape Signature" para la PARTE 1 (comparación con Hu y Zernike).

    Variante: FIRMA DE DISTANCIA AL CENTROIDE (centroid distance signature), la
    forma clásica de "shape signature":
        1. Se obtiene el contorno externo de la figura.
        2. Se remuestrea a N puntos equiespaciados (para que dos figuras sean
           comparables aunque tengan distinto número de píxeles de contorno).
        3. Se calcula la señal 1-D:  r(k) = || p_k − centroide ||
        4. Se aplica la FFT a r(k).
        5. Se toman los 13 primeros componentes de frecuencia:
              - |F(0)| (componente DC) -> se usa para NORMALIZAR (invarianza a
                escala).
              - los 12 siguientes |F(1)|..|F(12)| / |F(0)| -> descriptor.

    Invarianzas (Unidad 3.2):
        - Traslación: r(k) se mide respecto al centroide.
        - Rotación / punto de inicio: una rotación o un cambio del punto inicial
          equivalen a un desplazamiento circular de r(k); usar la MAGNITUD de la
          FFT elimina la fase, logrando invarianza.
        - Escala: división por el componente DC |F(0)|.

    Devuelve un vector de 12 valores (descriptor).
"""
import cv2
import numpy as np

from ...utils import config


def resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """
    Remuestrea un contorno cerrado (N,2) a 'n_points' puntos equiespaciados a lo
    largo de su perímetro (interpolación lineal por longitud de arco).
    """
    pts = contour.astype(np.float64)
    # Distancias acumuladas a lo largo del contorno (cerrado).
    closed = np.vstack([pts, pts[0]])
    deltas = np.diff(closed, axis=0)
    seg_len = np.sqrt((deltas ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        # Contorno degenerado: replicar el primer punto.
        return np.repeat(pts[:1], n_points, axis=0)

    target = np.linspace(0, total, n_points, endpoint=False)
    x = np.interp(target, cum, closed[:, 0])
    y = np.interp(target, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def centroid_distance_signature(contour: np.ndarray, n_points: int) -> np.ndarray:
    """Calcula r(k) = distancia de cada punto remuestreado al centroide."""
    pts = resample_contour(contour, n_points)
    centroid = pts.mean(axis=0)
    diff = pts - centroid
    return np.sqrt((diff ** 2).sum(axis=1))


def shape_signature_descriptor(contour: np.ndarray,
                               n_harmonics: int = config.SHAPE_SIG_N_HARMONICS,
                               n_points: int = 256) -> np.ndarray:
    """
    Descriptor Shape Signature (12 valores) a partir de un contorno (N,2).

    n_harmonics = 13  ->  F(0) normaliza + F(1..12) descriptor.
    """
    r = centroid_distance_signature(contour, n_points)

    # FFT de la firma 1-D y su magnitud.
    F = np.fft.fft(r)
    mag = np.abs(F)

    dc = mag[0] if mag[0] > 1e-12 else 1.0     # componente DC (|F(0)|)
    # Tomamos F(1)..F(n_harmonics-1) y normalizamos por DC.
    descriptor = mag[1:n_harmonics] / dc
    return descriptor.astype(np.float64)


def shape_signature_feature_names(n_harmonics: int = config.SHAPE_SIG_N_HARMONICS):
    return [f"SS_F{k}" for k in range(1, n_harmonics)]
