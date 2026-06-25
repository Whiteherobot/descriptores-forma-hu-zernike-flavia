"""
Ruta:           src/augmentation/rotation.py
Responsabilidad:
    Generar rotaciones aleatorias de 0º a 360º para cada imagen (Parte 1).

    Sirve para verificar EXPERIMENTALMENTE la invarianza a la rotación que la
    teoría atribuye a los momentos de Hu y Zernike y a la firma espectral.
    Se usa OpenCV (getRotationMatrix2D + warpAffine), equivalente sencillo y
    explicable de Albumentations ("o una similar", según la guía).

    Unidad: 3.1 (invarianza a la rotación de los momentos).
"""
import cv2
import numpy as np


def rotate(gray: np.ndarray, angle_deg: float,
           border_value: int = 255) -> np.ndarray:
    """
    Rota la imagen 'angle_deg' grados alrededor de su centro, expandiendo el
    lienzo para no recortar la figura. El fondo se rellena con 255 (papel claro)
    para mantener la convención tinta-oscura / fondo-claro.
    """
    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Calcular el nuevo tamaño del lienzo tras la rotación.
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(gray, M, (new_w, new_h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)


def random_rotation(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica una rotación aleatoria uniforme en [0, 360)."""
    angle = float(rng.uniform(0.0, 360.0))
    return rotate(gray, angle)
