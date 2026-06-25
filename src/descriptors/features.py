"""
Ruta:           src/descriptors/features.py
Responsabilidad:
    Orquestar la extracción de los TRES descriptores de la Parte 1 a partir de
    una imagen en gris, reutilizando el preprocesamiento común. Devuelve un
    diccionario { 'hu': vec, 'zernike': vec, 'shape_signature': vec } o None si
    la figura no pudo segmentarse.

    Mantener esta fachada en un solo lugar evita duplicar el preprocesamiento en
    el runner y el notebook.
"""
from typing import Dict, Optional

import numpy as np

from ..preprocessing.segmentation import filled_shape_mask, preprocessed_contour
from .hu.hu_moments import hu_moments
from .zernike.zernike_moments import zernike_moments
from .shape_signature.shape_signature import shape_signature_descriptor


def extract_all(gray: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """
    Extrae los descriptores Hu, Zernike y Shape Signature de una imagen en gris.
    Devuelve None si no se detecta figura (contorno/máscara vacíos).
    """
    mask = filled_shape_mask(gray)        # región rellena (Hu, Zernike)
    contour = preprocessed_contour(gray)  # contorno externo (Shape Signature)
    if mask is None or contour is None or len(contour) < 5:
        return None

    return {
        "hu": hu_moments(mask),
        "zernike": zernike_moments(mask),
        "shape_signature": shape_signature_descriptor(contour),
    }


DESCRIPTOR_NAMES = ["hu", "zernike", "shape_signature"]
