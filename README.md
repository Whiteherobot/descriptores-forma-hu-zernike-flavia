# Visión por Computador — Práctica 3.1 / 3.2

**Análisis comparativo de robustez (Hu vs Zernike vs Shape Signature)** y
**reconocimiento de hojas con Firma Espectral (FFT de coordenadas complejas)**.

- **Universidad Politécnica Salesiana** · Docente: Ing. Vladimir Robles Bykbaev
- **Estudiante:** Lata Zambrano Michael Israel — **Grupo G3**
- **Clases asignadas (Flavia, Tabla 2):** `14, 15, 16, 17, 18, 19, 20`
- **Parámetros personalizados (pista):** umbral **T = 130**, KNN **K = 7**
- Restricción: **ML clásico** (OpenCV / NumPy / scikit-learn). **Sin Deep Learning.**

---

## Estructura

```
vision_project/
├── data/
│   ├── ups_writing_skills/all-images/{circle,square,triangle}/   (Parte 1, incluido)
│   └── flavia/                      (Parte 2: colocar all.csv + imágenes)
├── notebooks/part1_robustez_hu_zernike.ipynb     (cuaderno estilo Kaggle)
├── src/
│   ├── preprocessing/   segmentation.py (Parte 1), leaf_segmentation.py (Parte 2)
│   ├── augmentation/    noise.py, rotation.py, pipeline.py
│   ├── descriptors/     hu/, zernike/, shape_signature/, features.py
│   ├── classifiers/     classifier.py (KNN K=7 / SVM)
│   ├── evaluation/      metrics.py (matriz de confusión, robustez, PCA)
│   └── utils/           config.py, io_utils.py
├── cpp/                 núcleo C++ nativo (Parte 2) + CLI de validación
├── mobile/             app Android (JNI + OpenCV) que usa el MISMO core C++
├── scripts/            run_part1.py, run_part2.py, build_notebook_part1.py
├── results/            figuras y métricas generadas
├── reports/            informes Parte 1 y Parte 2
└── requirements.txt
```

## Conceptos y unidades

| Concepto | Unidad | Dónde |
|---|---|---|
| Momentos de Hu (7 + log) | 3.1 | `src/descriptors/hu/hu_moments.py` |
| Momentos de Zernike (n=8, desde cero) | 3.1 | `src/descriptors/zernike/zernike_moments.py` |
| Shape Signature (distancia al centroide + FFT) | 3.2 | `src/descriptors/shape_signature/shape_signature.py` |
| Descriptor de Fourier (coordenadas complejas) | 3.2 | `fourier_complex.py` y `cpp/` |
| Distancia Euclídea (clasificación) | 3.1 | `fourier_complex.py`, `cpp/` |
| Segmentación / centroide / momentos | 3.1 | `src/preprocessing/` |

---

## Parte 1 — Robustez Hu vs Zernike vs Shape Signature (Python)

```bash
python -m scripts.run_part1 --full      # dataset completo (genera results/part1/)
python -m scripts.run_part1             # modo rápido (subconjunto)
```
Genera matrices de confusión por nivel de ruido, curvas de robustez (accuracy vs
ruido), PCA y `summary.json`. El cuaderno equivalente está en `notebooks/`.

## Parte 2 — Reconocimiento de hojas (C++ nativo + app Android)

1. **Datos:** ya descargados en `data/flavia/` (Kaggle `gauravneupane/flavia-dataset`:
   `all.csv` + `images/` con 1907 hojas).
2. **Referencia Python** (genera matriz de confusión, accuracy y `model.csv`):
   ```bash
   python -m scripts.run_part2
   ```
3. **Núcleo C++** (mismo algoritmo, validable por CLI):
   ```bash
   cd cpp && cmake -B build && cmake --build build -j
   ./build/selftest                       # prueba sin dataset (formas sintéticas)
   ./build/flavia_eval ../data/flavia model.csv
   ```
4. **App Android:** ver `mobile/README_MOBILE.md` (integra el dibujo manual y el
   mismo core C++ vía JNI). Copia `results/part2/model.csv` a
   `mobile/app/src/main/assets/model.csv`.

---

## Resultados (resumen)

- **Parte 1:** Zernike y Shape Signature mantienen accuracy alta y estable frente
  al ruido (~0.80–0.89); **Hu es el descriptor más débil y el que más se degrada**
  (~0.60–0.65). Confirma la *superioridad de la ortogonalidad* de Zernike.
- **Parte 2:** sobre el **Flavia Dataset real** (clases 14–20, 442 hojas)
  **Accuracy = 93.5 %** — idéntico en Python y en el núcleo **C++**. El descriptor
  de Fourier de coordenadas complejas es invariante a rotación/escala/traslación;
  las únicas confusiones (6) son entre especies lanceoladas casi idénticas (14↔15).

Detalle en `reports/`.
