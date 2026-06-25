# App Android — Parte 2 (Shape Signature / Coordenadas Complejas)

App nativa que reconoce la **especie de una hoja** (clases asignadas al G3:
**14–20**) usando el **Descriptor de Fourier de coordenadas complejas** calculado
en una **librería nativa de C++** (la misma que se valida en `cpp/`).

## Arquitectura

```
UI (Java)                         Núcleo C++ (JNI)
─────────                         ────────────────
MainActivity ── botón ──► NativeBridge.nativeClassify(matAddr)
DrawView (lienzo)                 │
   │ Bitmap                       ▼
   └─► OpenCV Utils.bitmapToMat ─► native_lib.cpp
                                   └─► shape_descriptor.cpp  (cv::dft, |F|/|F(1)|,
                                        distancia Euclídea, KNN K=7)
```

- **Una sola fuente de verdad:** `app/src/main/cpp/CMakeLists.txt` compila el
  archivo compartido `../../../../../cpp/src/shape_descriptor.cpp`. El algoritmo
  del móvil es **idéntico** al de la CLI de escritorio ya validada.
- El **modelo** (`assets/model.csv`) son los descriptores de entrenamiento
  (80 % por clase) generados por `scripts/run_part2.py` o por `flavia_eval`.
  **Ya está empaquetado** (350 muestras de las clases 14–20).
- **OpenCV** se resuelve automáticamente por **Maven Central** (`org.opencv:opencv`
  con **Prefab**): incluye los headers y las librerías nativas, por lo que **NO
  necesitas descargar el OpenCV Android SDK** ni configurar `opencv.dir`.

## Requisitos
- Android Studio (Giraffe o superior). Android Studio descargará solo el **NDK**,
  **CMake**, el **platform android-34** y la distribución de **Gradle** si faltan.

## Pasos para compilar e instalar en la tablet
1. **Activa el modo desarrollador** en la tablet: Ajustes → *Acerca del teléfono* →
   pulsa 7 veces *Número de compilación*; luego Ajustes → *Opciones de
   desarrollador* → activa **Depuración por USB**. Conecta la tablet por USB y
   acepta el diálogo *Permitir depuración*.
2. Abre **Android Studio** → *Open* → selecciona la carpeta `mobile/`.
3. Espera el *Gradle Sync* (la primera vez descarga Gradle, AGP y el AAR de
   OpenCV ~115 MB). Si pide instalar NDK/CMake/SDK 34, acepta.
4. Arriba, en el selector de dispositivos, elige tu **tablet**; pulsa **Run ▶**.
   La app se compila, instala y abre en la tablet.

> El modelo ya está en `assets/model.csv`. Si reentrenas con otro subconjunto,
> regénéralo con `python -m scripts.run_part2` y copia
> `results/part2/model.csv` sobre `mobile/app/src/main/assets/model.csv`.

### Alternativa por línea de comandos
```bash
cd mobile
./gradlew installDebug      # compila e instala en la tablet conectada (adb)
```

## Uso
1. Dibuja la hoja en el área blanca.
2. Pulsa **Clasificar forma**.
3. La app muestra **ESPECIE CLASIFICADA: N** (14–20) y los primeros valores del
   descriptor de Fourier.

> Nota: la app integra el dibujo manual (como exige la guía). Para clasificar
> fotos de hojas reales puede ampliarse `MainActivity` para tomar la imagen de la
> cámara y pasarla al mismo `nativeClassify` (el núcleo C++ ya soporta fotos vía
> `describeImage(..., useAdaptiveThreshold=false)`).
