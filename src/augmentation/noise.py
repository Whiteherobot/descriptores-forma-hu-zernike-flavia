"""
Ruta:           src/augmentation/noise.py
Responsabilidad:
    Implementar la "Inyección de Ruido" pedida en la Parte 1 de la guía:
        - Ruido Gaussiano  (aditivo)
        - Ruido Salt & Pepper (impulsivo)
    cada uno con 3 niveles de intensidad: Bajo, Medio y Alto.

    El objetivo académico es estudiar cómo el ruido degrada las características
    globales (momentos) y comparar la robustez de Hu vs Zernike vs Shape
    Signature. (Unidad 3 - Análisis de degradación.)

    Se implementa con NumPy puro (sin librerías externas) para que sea totalmente
    explicable.
"""
import numpy as np

from ..utils import config


def add_gaussian_noise(gray: np.ndarray, sigma_rel: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Ruido Gaussiano aditivo. sigma_rel es la desviación estándar relativa a 255.
    g'(x,y) = g(x,y) + N(0, sigma)   con recorte a [0,255].
    """
    sigma = sigma_rel * 255.0
    noise = rng.normal(0.0, sigma, size=gray.shape)
    out = gray.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(gray: np.ndarray, amount: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Ruido impulsivo Salt & Pepper. 'amount' es la fracción total de píxeles
    alterados (mitad sal=255, mitad pimienta=0).
    """
    out = gray.copy()
    n = gray.size
    n_affected = int(amount * n)
    if n_affected == 0:
        return out

    flat = out.reshape(-1)
    idx = rng.choice(n, size=n_affected, replace=False)
    half = n_affected // 2
    flat[idx[:half]] = 255   # sal
    flat[idx[half:]] = 0     # pimienta
    return flat.reshape(gray.shape)


def apply_noise(gray: np.ndarray, noise_type: str, level: str,
                rng: np.random.Generator) -> np.ndarray:
    """
    Despachador genérico.
        noise_type : 'gaussian' | 'salt_pepper'
        level      : 'bajo' | 'medio' | 'alto'
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(gray, config.GAUSSIAN_LEVELS[level], rng)
    if noise_type == "salt_pepper":
        return add_salt_pepper(gray, config.SALT_PEPPER_LEVELS[level], rng)
    raise ValueError(f"Tipo de ruido desconocido: {noise_type}")
