# Validación contra la guía de prácticas (checklist)

## Parte 1 — Cuaderno de Kaggle (Hu vs Zernike vs Shape Signature)

| Requisito de la guía | Estado | Dónde |
|---|---|---|
| Clonar/usar cuaderno "UPS-Writing-Skills" | ✅ | `notebooks/part1_robustez_hu_zernike.ipynb` |
| Cargar clases Circle / Triangle / Rectangle(Square) | ✅ | `src/utils/io_utils.py`, `data/ups_writing_skills/` |
| Función de inyección de ruido Gaussian + Salt & Pepper, 3 niveles | ✅ | `src/augmentation/noise.py` |
| Rotaciones aleatorias 0–360° | ✅ | `src/augmentation/rotation.py` |
| Hu: 7 momentos + transformación log | ✅ | `src/descriptors/hu/hu_moments.py` |
| Zernike: grado n = 8 | ✅ | `src/descriptors/zernike/zernike_moments.py` (n=8, desde cero) |
| Shape Signature: 13 componentes (DC + 12) | ✅ | `src/descriptors/shape_signature/shape_signature.py` |
| Clasificador simple (KNN o SVM) | ✅ | `src/classifiers/classifier.py` (KNN K=7 / SVM) |
| Matriz de confusión por nivel de ruido | ✅ | `results/part1/cm_*.png` |
| Métricas de precisión (accuracy) | ✅ | `results/part1/summary.json`, curvas de robustez |
| PCA para ver mezcla de clusters | ✅ | `results/part1/pca_zernike_limpio.png` |
| Filtro de mediana ante Salt & Pepper | ✅ | `src/preprocessing/segmentation.py` |

## Parte 2 — App móvil (Shape Signature / coordenadas complejas)

| Requisito de la guía | Estado | Dónde |
|---|---|---|
| Librería nativa **C++** que realiza los cálculos | ✅ | `cpp/src/shape_descriptor.cpp` (compartida con Android) |
| Referencia al corpus Flavia, clases asignadas 14–20 | ✅ | `config.FLAVIA_CLASSES`, `scripts/run_part2.py`, `cpp/apps/main.cpp` |
| App con área de dibujo + botón "Clasificar forma" | ✅ | `mobile/` (DrawView, MainActivity) |
| Muestra la etiqueta de especie (0–31; aquí 14–20) | ✅ | `native_lib.cpp` → "ESPECIE CLASIFICADA: N" |
| Preprocesar: reducir ruido (Gaussiano) | ✅ | `extractLargestContour` (GaussianBlur) |
| Detectar contornos (`findContours`) | ✅ | `shape_descriptor.cpp`, `leaf_segmentation.py` |
| Firma + FFT | ✅ | `fourierDescriptor` (cv::dft) / `fourier_complex.py` |
| s(n) = (x−xc) + j(y−yc) (coordenadas complejas) | ✅ | `fourierDescriptor`, `fourier_complex.py` |
| Segmentación con umbral adaptativo | ✅ | `extractLargestContour(useAdaptiveThreshold=true)` |
| `cv::dft` con `DFT_COMPLEX_OUTPUT` | ✅ | `shape_descriptor.cpp` |
| Normalizar por \|F(1)\| y descartar fase | ✅ | `fourierDescriptor` |
| Split train 80 % / test 20 % por clase | ✅ | `run_part2.py`, `main.cpp` (`TRAIN_RATIO`) |
| Distancia Euclídea (vista con Hu) | ✅ | `euclideanDistance` / `euclidean_distance` |
| Reporte: código, matriz de confusión, precisión, formas confundidas | ✅ | `reports/informe_parte2.md`, `results/part2/` |
| Sugerencia: Albumentations o similar | ✅ | rotaciones/ruido propios (equivalente explicable) |

## Restricciones

| Restricción | Cumplimiento |
|---|---|
| Sin Deep Learning / redes neuronales | ✅ sólo OpenCV/NumPy/SciPy/scikit-learn |
| Sólo técnicas vistas en clase | ✅ Hu, Zernike, FFT, KNN, distancia Euclídea, PCA |
| Código explicable y defendible | ✅ módulos documentados con su unidad |

## Estado de la Parte 2 (Flavia real)
- ✅ Dataset descargado (`gauravneupane/flavia-dataset`) y enlazado en `data/flavia/`.
- ✅ Ejecutado: **Accuracy = 93.5 %** (Python y C++ idénticos), matriz de confusión
  y ejemplos de hojas confundidas en `results/part2/`.
- ✅ Modelo entrenado empaquetado en `mobile/app/src/main/assets/model.csv`.

## Estado de la App Android (compilada y ejecutada en dispositivo real)
- ✅ OpenCV se resuelve por **Maven/Prefab** (`org.opencv:opencv:4.11.0`); NO se
  necesita descargar el OpenCV Android SDK ni `opencv.dir`.
- ✅ **APK compilado e instalado** en tablet física (modelo 9185W, Android 15) vía
  `./gradlew installDebug` (JBR de Android Studio en `/opt/android-studio/jbr`).
- ✅ **Dibujo → "Clasificar forma" → "ESPECIE CLASIFICADA: N"** funciona en vivo.
- Correcciones aplicadas para que corriera en el móvil:
  1. `shape_descriptor.cpp`: la imagen de Android llega en **4 canales (RGBA)**; se
     añadió `COLOR_BGRA2GRAY` (antes pasaba 4 canales a `adaptiveThreshold` → crash).
  2. `native_lib.cpp`: `try/catch` en `nativeClassify` (mensaje en pantalla en vez
     de cerrarse ante cualquier excepción de OpenCV).
  3. `DrawView.getBitmap`: copia del bitmap como **mutable** (`copy(..., true)`); el
     `Canvas` exige bitmap mutable (antes `IllegalStateException`).
  4. `build.gradle`: `-DANDROID_STL=c++_shared` (OpenCV prefab exige STL compartida).
