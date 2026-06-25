"""
Ruta:           src/utils/config.py
Responsabilidad:
    Centralizar TODOS los parámetros del proyecto en un único lugar para que el
    sistema sea reproducible y fácil de defender en la sustentación.

    Aquí se fijan los parámetros personalizados entregados en la "pista" del
    docente para el estudiante LATA ZAMBRANO MICHAEL ISRAEL (Grupo G3):

        T = 130   -> Umbral fijo de binarización (segmentación de la tinta)
        K = 7     -> Número de vecinos del clasificador KNN

    Unidad de la asignatura:
        - Unidad 3.1 (Momentos de Hu / Zernike) y 3.2 (Shape Signature + FFT).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base del proyecto (se derivan de la ubicación de este archivo)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]      # .../vision_project/src
PROJECT_ROOT = SRC_DIR.parent                       # .../vision_project

DATA_DIR = PROJECT_ROOT / "data"
UPS_DIR = DATA_DIR / "ups_writing_skills" / "all-images"   # circle / square / triangle
FLAVIA_DIR = DATA_DIR / "flavia"                            # all.csv + imágenes (lo coloca el usuario)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PART1 = RESULTS_DIR / "part1"
RESULTS_PART2 = RESULTS_DIR / "part2"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Parámetros PERSONALIZADOS (pista del docente)
# ---------------------------------------------------------------------------
T_THRESHOLD = 130   # umbral fijo de binarización  (segmentación)
KNN_K = 7           # vecinos del clasificador KNN

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
# Tamaño al que se normaliza cada figura antes de extraer descriptores.
# Un tamaño común evita que la escala del papel afecte a Zernike (que necesita
# un círculo unitario bien definido) y acelera el cálculo.
NORMALIZE_SIZE = 256
MEDIAN_KSIZE = 3    # filtro de mediana para atenuar Salt & Pepper (Recomendación de la guía)

# ---------------------------------------------------------------------------
# PARTE 1 - Aumento de datos (UPS-Writing-Skills)
# ---------------------------------------------------------------------------
UPS_CLASSES = ["circle", "square", "triangle"]   # círculo, cuadrado/rectángulo, triángulo

# Tres niveles de intensidad de ruido (Bajo, Medio, Alto), como pide la guía.
GAUSSIAN_LEVELS = {"bajo": 0.02, "medio": 0.06, "alto": 0.12}      # sigma relativo (0..1)
SALT_PEPPER_LEVELS = {"bajo": 0.02, "medio": 0.06, "alto": 0.12}   # proporción de píxeles afectados

# Variantes aumentadas por imagen (cada una con una rotación aleatoria 0..360).
AUG_VARIANTS_PER_IMAGE = 6

# ---------------------------------------------------------------------------
# PARTE 1 - Descriptores
# ---------------------------------------------------------------------------
ZERNIKE_DEGREE = 8          # grado del polinomio n=8 (capturar detalles finos)
SHAPE_SIG_N_HARMONICS = 13  # 1 componente DC (normaliza) + 12 valores como descriptor

# ---------------------------------------------------------------------------
# PARTE 2 - Flavia (Shape Signature con coordenadas complejas + FFT)
# ---------------------------------------------------------------------------
# Clases asignadas al Grupo G3 (LATA ZAMBRANO / ROMERO ARMIJOS) según la Tabla 2.
FLAVIA_CLASSES = [14, 15, 16, 17, 18, 19, 20]
FLAVIA_CSV_NAME = "all.csv"        # columnas: count,id,y
TRAIN_RATIO = 0.80                 # 80% train / 20% test por clase
# Número de coeficientes de Fourier (armónicos) usados como descriptor de la firma.
# Se descartan F(0) (traslación) y se normaliza por |F(1)| (escala).
FOURIER_DESCRIPTOR_SIZE = 32
CONTOUR_RESAMPLE_POINTS = 128      # remuestreo del contorno a N puntos para que la FFT sea comparable
