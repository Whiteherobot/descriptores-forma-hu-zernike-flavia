"""
Ruta:           src/descriptors/zernike/zernike_moments.py
Responsabilidad:
    Implementar DESDE CERO (sólo NumPy) el cálculo de los Momentos de Zernike,
    siguiendo las fórmulas del material de clase ("Introducción a los momentos
    de Zernike"). No se usa ninguna librería de momentos (ni mahotas), para que
    todo el procedimiento sea explicable en la sustentación.

    Fundamento (Unidad 3.1):
        Polinomio radial:
            R_nm(ρ) = Σ_{s=0}^{(n-|m|)/2} (-1)^s · (n-s)! /
                      [ s! · ((n+|m|)/2 - s)! · ((n-|m|)/2 - s)! ] · ρ^(n-2s)

        Función base compleja:
            V_nm(x,y) = R_nm(ρ) · exp(j·m·θ)

        Momento de Zernike (proyección sobre la base, dentro del círculo unitario):
            A_nm = (n+1)/π · Σ_{x²+y² ≤ 1} I(x,y) · conj(V_nm(x,y))

    Restricciones de los índices (del PDF):
        |m| ≤ n   y   (n − |m|) par.

    Invarianza a la rotación:
        Una rotación de la figura sólo cambia la FASE de A_nm, por lo que el
        descriptor usa las MAGNITUDES |A_nm| (invariantes a rotación). Como la
        figura se centra y se escala a un tamaño fijo antes de proyectar, el
        descriptor es además invariante a traslación y escala.

    Grado usado: n = 8 (config.ZERNIKE_DEGREE) -> 25 momentos.
"""
from functools import lru_cache
from math import factorial, pi
from typing import List, Tuple

import numpy as np

from ...utils import config


def _valid_nm(max_degree: int) -> List[Tuple[int, int]]:
    """
    Lista de pares (n, m) válidos con m >= 0 (se usa m>=0 porque
    |A_{n,-m}| = |A_{n,m}|, evitando redundancia), |m|<=n y (n-m) par.
    """
    pairs = []
    for n in range(max_degree + 1):
        for m in range(0, n + 1):
            if (n - m) % 2 == 0:
                pairs.append((n, m))
    return pairs


def _radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Evalúa R_nm(ρ) (vectorizado sobre el arreglo rho)."""
    m = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s) * factorial(n - s)
        c /= (factorial(s)
              * factorial((n + m) // 2 - s)
              * factorial((n - m) // 2 - s))
        R += c * rho ** (n - 2 * s)
    return R


@lru_cache(maxsize=4)
def _build_basis(size: int, max_degree: int):
    """
    Precalcula (y cachea) la base de Zernike para imágenes de 'size x size'.

    Se asume la figura CENTRADA en el lienzo (el preprocesamiento ya la centra)
    y un círculo unitario inscrito de radio = size/2.

    Devuelve:
        pairs : lista de (n,m)
        basis : arreglo complejo (len(pairs), K) con conj(V_nm) SÓLO en los
                píxeles dentro del círculo unitario (máscara aplanada).
        mask_disk : índices booleanos de los píxeles dentro del disco.
        norm  : factores (n+1)/π por cada par.
    """
    # Coordenadas normalizadas: centro del lienzo -> (0,0); borde -> radio 1.
    radius = size / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    x = (xs - radius) / radius
    y = (ys - radius) / radius

    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    disk = rho <= 1.0   # sólo el interior del círculo unitario

    rho_d = rho[disk]
    theta_d = theta[disk]

    pairs = _valid_nm(max_degree)
    basis = np.empty((len(pairs), rho_d.size), dtype=np.complex128)
    norm = np.empty(len(pairs), dtype=np.float64)

    for i, (n, m) in enumerate(pairs):
        R = _radial_poly(n, m, rho_d)
        V = R * np.exp(1j * m * theta_d)     # V_nm
        basis[i] = np.conj(V)                # se proyecta con conj(V)
        norm[i] = (n + 1) / pi

    return pairs, basis, disk, norm


def zernike_moments(mask: np.ndarray,
                    max_degree: int = config.ZERNIKE_DEGREE) -> np.ndarray:
    """
    Calcula el descriptor de Zernike (magnitudes |A_nm|) de una máscara binaria.

    Entrada:
        mask : máscara binaria cuadrada (size x size) uint8 {0,255}, figura
               centrada (salida de preprocessing.filled_shape_mask).
    Salida:
        vector de magnitudes |A_nm| (float64), longitud = nº de pares (n,m).
    """
    size = mask.shape[0]
    if mask.shape[0] != mask.shape[1]:
        raise ValueError("La máscara debe ser cuadrada para Zernike.")

    pairs, basis, disk, norm = _build_basis(size, max_degree)

    # Imagen normalizada a {0,1} dentro del disco.
    img = (mask > 0).astype(np.float64)[disk]

    # Proyección: A_nm = (n+1)/π · Σ I · conj(V_nm)
    A = basis @ img            # producto vectorizado (suma sobre píxeles)
    A = A * norm
    return np.abs(A)


def zernike_feature_names(max_degree: int = config.ZERNIKE_DEGREE) -> List[str]:
    """Nombres legibles 'Z_n_m' para las tablas del reporte."""
    return [f"Z{n}_{m}" for (n, m) in _valid_nm(max_degree)]
