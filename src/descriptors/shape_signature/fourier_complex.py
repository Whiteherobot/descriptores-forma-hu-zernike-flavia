"""
Ruta:           src/descriptors/shape_signature/fourier_complex.py
Responsabilidad:
    Descriptor de Fourier con la VARIANTE DE COORDENADAS COMPLEJAS, exigido en
    la PARTE 2 (Flavia Dataset). Es la referencia en Python del MISMO algoritmo
    implementado en C++ (mobile/.../shape_descriptor.cpp): ambos deben producir
    el mismo vector.

    Fundamento (guía Parte 2 / Unidad 3.2):
        El contorno se trata como una señal en el plano complejo:
            s(n) = ( x(n) − xc ) + j ( y(n) − yc )
        donde (xc, yc) es el centroide. Esta representación conserva la relación
        espacial completa entre los puntos del contorno.

    Flujo:
        1. Remuestrear el contorno a N puntos (config.CONTOUR_RESAMPLE_POINTS)
           para que las firmas sean comparables.
        2. Construir s(n) (coordenadas complejas respecto al centroide).
        3. DFT:  F(k) = FFT(s(n)).
        4. Normalización:
              - Se descarta F(0)        -> invarianza a TRASLACIÓN.
              - Se divide por |F(1)|    -> invarianza a ESCALA.
              - Se usan MAGNITUDES      -> invarianza a ROTACIÓN (se descarta la fase).
        5. Se toman los FOURIER_DESCRIPTOR_SIZE coeficientes de más baja
           frecuencia (mitad positivos, mitad negativos), que describen la forma
           global sin el ruido de alta frecuencia (Recomendación de la guía).
"""
import numpy as np

from ...utils import config
# Reutilizamos el remuestreo por longitud de arco de la firma de distancia.
from .shape_signature import resample_contour


def fourier_descriptor(contour: np.ndarray,
                       n_points: int = config.CONTOUR_RESAMPLE_POINTS,
                       n_descriptors: int = config.FOURIER_DESCRIPTOR_SIZE) -> np.ndarray:
    """
    Calcula el descriptor de Fourier (coordenadas complejas) de un contorno.

    Entrada:
        contour : arreglo (N,2) con los puntos del contorno externo.
    Salida:
        vector real de longitud n_descriptors (magnitudes normalizadas).
    """
    pts = resample_contour(contour, n_points)

    # Orientación consistente (antihoraria): cv2.findContours puede devolver el
    # contorno en sentido horario, lo que coloca el armónico fundamental en
    # F(N-1) en vez de F(1). Forzamos área con signo positiva para que F(1) sea
    # SIEMPRE la frecuencia fundamental y la normalización por |F(1)| sea estable.
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area < 0:
        pts = pts[::-1]

    xc, yc = pts.mean(axis=0)

    # Señal compleja s(n).
    s = (pts[:, 0] - xc) + 1j * (pts[:, 1] - yc)

    # DFT.
    F = np.fft.fft(s)
    mag = np.abs(F)

    # |F(1)| = primer armónico (frecuencia fundamental) -> normaliza escala.
    # Por robustez se toma el mayor de los dos armónicos fundamentales.
    f1 = max(mag[1], mag[n_points - 1])
    if f1 < 1e-12:
        f1 = 1.0

    half = n_descriptors // 2
    # Coeficientes de baja frecuencia: positivos F(1..half) y negativos F(N-half..N-1).
    # (F(0) se descarta: corresponde al centroide / traslación.)
    pos = mag[1:half + 1]
    neg = mag[n_points - half:n_points]
    desc = np.concatenate([pos, neg]) / f1
    return desc.astype(np.float64)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia Euclídea entre dos descriptores (vista en clase en el tema de los
    momentos de Hu). Se usa para clasificar por vecino más cercano en la Parte 2.
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))
