# Informe — Parte 2
## Reconocimiento de hojas (Flavia) con Descriptor de Fourier de Coordenadas Complejas

**Estudiante:** Lata Zambrano Michael Israel (G3) · **Clases asignadas:**
14, 15, 16, 17, 18, 19, 20 · **Parámetros:** K = 7, train/test 80/20.

---

### 1. Metodología (Unidad 3.2)

1. **Preprocesamiento.** Reducción de ruido con **filtro Gaussiano**; binarización
   (Otsu para fotos de hojas; **umbral adaptativo** para el dibujo en vivo del
   móvil); cierre morfológico; **`cv::findContours`** con `RETR_EXTERNAL` para el
   contorno externo más grande.
2. **Variante de coordenadas complejas.** El contorno se trata como una señal en
   el plano complejo:

   ```
   s(n) = (x(n) − xc) + j (y(n) − yc)
   ```

   donde `(xc, yc)` es el centroide. Conserva la relación espacial completa.
3. **DFT (`cv::dft`).** Se aplica la transformada a `s(n)`.
4. **Normalización (invarianzas):**
   - Se descarta **F(0)** → invarianza a **traslación**.
   - Se divide por **|F(1)|** (primer armónico) → invarianza a **escala**.
   - Se usan **magnitudes** (se descarta la fase) → invarianza a **rotación**.
   - Se fuerza la **orientación antihoraria** del contorno para que F(1) sea
     siempre la frecuencia fundamental (estabilidad numérica).
   - Descriptor: 32 coeficientes de baja frecuencia (16 positivos + 16 negativos).
5. **Clasificación.** **Distancia Euclídea** (vista en clase con Hu) + KNN K = 7
   contra los descriptores de entrenamiento (80 % de cada clase).

### 2. Implementación

- **Núcleo C++ nativo** (`cpp/src/shape_descriptor.cpp`): segmentación,
  descriptor de Fourier (`cv::dft`), distancia Euclídea y KNN.
- **CLI de validación** (`cpp/apps/`): `selftest` (sin dataset) y `flavia_eval`
  (entrena/evalúa sobre Flavia, imprime accuracy y matriz de confusión, exporta
  `model.csv`).
- **Referencia Python** (`scripts/run_part2.py`,
  `src/descriptors/shape_signature/fourier_complex.py`): mismo algoritmo, produce
  el **mismo descriptor** que el C++.
- **App Android** (`mobile/`): reutiliza el mismo `.cpp` vía **JNI**; el usuario
  dibuja la hoja y la app muestra la **especie (14–20)** y el descriptor.

### 3. Resultados sobre el Flavia Dataset REAL (clases 14–20)

Dataset: `gauravneupane/flavia-dataset` (1907 hojas, 32 clases). Para el G3 se
usan las clases 14–20 → **442 hojas** (60, 56, 77, 62, 61, 66, 60). Segmentación
exitosa en las 442 (0 fallos). Split 80/20 por clase: **train = 350, test = 92**.

> **Accuracy = 0.9348 (93.5 %)** — idéntico en la referencia Python y en el núcleo
> C++ (`flavia_eval`).

**Matriz de confusión** (`results/part2/confusion_matrix_part2.png`):

```
        14   15   16   17   18   19   20
  14     9    3    0    0    0    0    0
  15     2   10    0    0    0    0    0
  16     0    0   16    0    0    0    0
  17     0    0    0   13    0    0    0
  18     0    1    0    0   12    0    0
  19     0    0    0    0    0   14    0
  20     0    0    0    0    0    0   12
```

Solo **6 hojas** mal clasificadas, todas entre las clases **14 ↔ 15** (y una
18→15). Como se ve en `results/part2/hojas_confundidas.png`, esas especies tienen
hojas **lanceoladas casi idénticas** en forma, por lo que su descriptor de
contorno es muy parecido — es una confusión esperable y justificable. Las clases
16, 17, 19 y 20 se clasifican perfectamente.

> Reproducción: `python -m scripts.run_part2`  ó
> `./cpp/build/flavia_eval data/flavia model.csv`.

### 4. Validación del algoritmo (formas sintéticas)

`./cpp/build/selftest` y la prueba equivalente en Python verifican, sobre formas
sintéticas, que el descriptor es **invariante**:

| Comparación | Distancia (C++) | Distancia (Python) |
|---|---|---|
| círculo vs mismo **rotado 47°** | 0.0006 | 0.0000 |
| círculo vs mismo **escala 1.6×** | 0.0004 | 0.0007 |
| círculo vs mismo **trasladado** | ~0 | 0.0000 |
| círculo vs **cuadrado** | 0.119 | 0.120 |
| círculo vs **triángulo** | 0.261 | 0.263 |

Las distancias entre variantes de la **misma** forma son ≈ 0 y mucho menores que
entre formas **distintas** → el descriptor separa clases. **El núcleo C++ y la
referencia Python coinciden numéricamente.**

> Validación de extremo a extremo: con un Flavia **sintético** de 7 clases, tanto
> `run_part2.py` como `flavia_eval` ejecutan el flujo completo (carga de
> `all.csv`, split 80/20, clasificación, matriz de confusión) y producen una
> matriz de confusión diagonal.

### 5. Cómo reproducir desde cero

```bash
# 0) Descargar el dataset (ya realizado en este equipo):
python -c "import kagglehub; print(kagglehub.dataset_download('gauravneupane/flavia-dataset'))"
#    y enlazarlo en data/flavia/  (all.csv + images/ -> carpeta Leaves)

# 1) Referencia Python  -> results/part2/ (matriz, accuracy, model.csv)
python -m scripts.run_part2
# 2) Núcleo C++ (mismo resultado)
cd cpp && cmake -B build && cmake --build build -j
./build/flavia_eval ../data/flavia model.csv
# 3) App: copiar el modelo a los assets (ya hecho)
cp results/part2/model.csv mobile/app/src/main/assets/model.csv
```

Artefactos generados en `results/part2/`: `confusion_matrix_part2.png`,
`hojas_confundidas.png` (ejemplos de formas que confunde) y `model.csv` (modelo
para la app). El **modelo ya está empaquetado** en
`mobile/app/src/main/assets/model.csv` (350 descriptores de entrenamiento).

### 6. Conclusión

La **FFT de coordenadas complejas** es una representación compacta y eficiente del
contorno, fácil de normalizar para lograr invarianza a escala y rotación sin alto
costo computacional — ideal para el entorno móvil. La implementación en una
**librería nativa C++** compartida garantiza que el cálculo del teléfono sea
idéntico al validado en escritorio.
