"""
Ruta:           src/augmentation/pipeline.py
Responsabilidad:
    Construir, a partir de las imágenes base, el conjunto AUMENTADO de la Parte 1:
        - Por cada imagen se generan varias variantes con ROTACIÓN aleatoria
          (0..360º), para verificar la invarianza a la rotación.
        - A cada variante se le inyecta RUIDO (Gaussiano o Salt & Pepper) del
          nivel indicado (o ninguno si el nivel es 'limpio').
    Luego extrae los tres descriptores de cada variante.

    Devuelve matrices de características listas para el clasificador.
"""
from typing import Dict, List, Tuple

import numpy as np

from ..descriptors.features import extract_all, DESCRIPTOR_NAMES
from ..utils import config
from .noise import apply_noise
from .rotation import random_rotation


def build_feature_set(images: List[np.ndarray],
                      labels: List[int],
                      noise_type: str,
                      level: str,
                      variants: int = config.AUG_VARIANTS_PER_IMAGE,
                      seed: int = config.RANDOM_SEED
                      ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Genera variantes aumentadas y extrae descriptores.

    Parámetros:
        images, labels : dataset base.
        noise_type     : 'gaussian' | 'salt_pepper'.
        level          : 'limpio' (sin ruido) | 'bajo' | 'medio' | 'alto'.
        variants       : nº de copias rotadas por imagen.

    Devuelve:
        X_by_desc : { 'hu': X_hu, 'zernike': X_z, 'shape_signature': X_ss }
                    cada X es (n_muestras, n_features).
        y         : etiquetas alineadas (n_muestras,).
        groups    : id de la imagen ORIGEN de cada muestra. Permite separar
                    train/test por imagen base (evita data leakage entre
                    variantes de la misma figura).
    """
    rng = np.random.default_rng(seed)
    feats: Dict[str, List[np.ndarray]] = {d: [] for d in DESCRIPTOR_NAMES}
    ys: List[int] = []
    groups: List[int] = []

    for gid, (img, lab) in enumerate(zip(images, labels)):
        for _ in range(variants):
            aug = random_rotation(img, rng)              # rotación 0..360
            if level != "limpio":
                aug = apply_noise(aug, noise_type, level, rng)  # ruido

            d = extract_all(aug)
            if d is None:
                continue   # figura no segmentable (ruido extremo): se descarta
            for name in DESCRIPTOR_NAMES:
                feats[name].append(d[name])
            ys.append(lab)
            groups.append(gid)

    X_by_desc = {name: np.asarray(v, dtype=np.float64) for name, v in feats.items()}
    return (X_by_desc,
            np.asarray(ys, dtype=np.int64),
            np.asarray(groups, dtype=np.int64))
