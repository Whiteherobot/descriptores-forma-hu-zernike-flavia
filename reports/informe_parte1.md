# Informe — Parte 1
## Análisis comparativo de robustez: Momentos de Hu vs Zernike vs Shape Signature

**Estudiante:** Lata Zambrano Michael Israel (G3) · **Dataset:** UPS-Writing-Skills
(`circle`, `square`, `triangle`) · **Parámetros:** T = 130, KNN K = 7.

---

### 1. Metodología

1. **Segmentación (Unidad 3.1).** Umbral fijo **T = 130** (`THRESH_BINARY_INV`)
   sobre la imagen suavizada con filtro de **mediana** (atenúa Salt & Pepper).
   Como las tres clases son **convexas**, la silueta sólida se obtiene con la
   **envolvente convexa** del trazo, robusta a trazos abiertos. La figura se
   centra en su **centroide** y se escala por su radio máximo (encuadre
   consistente en el disco unitario de Zernike).
2. **Aumento de datos.** Para cada imagen se generan variantes con **rotación
   aleatoria 0–360°** y se inyecta ruido **Gaussiano** y **Salt & Pepper** en 3
   niveles (Bajo/Medio/Alto).
3. **Descriptores.**
   - **Hu:** 7 momentos invariantes + transformación logarítmica.
   - **Zernike:** grado **n = 8** (25 magnitudes |A_nm|), implementado **desde
     cero** con el polinomio radial y la base compleja del material de clase.
   - **Shape Signature:** firma de **distancia al centroide** del contorno →
     **FFT** → 13 componentes (DC normaliza + 12 descriptor).
4. **Clasificación.** KNN (**K = 7**) con estandarización. La partición train/test
   es **por grupos** (las variantes de una misma imagen base no se reparten entre
   train y test) para evitar *data leakage*.
5. **Evaluación.** Matriz de confusión por nivel de ruido, accuracy, curvas de
   robustez y PCA.

### 2. Resultados (accuracy)

**Ruido Gaussiano**

| Descriptor | Limpio | Bajo | Medio | Alto |
|---|---|---|---|---|
| Hu | 0.645 | 0.623 | 0.606 | 0.651 |
| Zernike | 0.822 | 0.822 | 0.814 | 0.842 |
| Shape Signature | 0.830 | 0.792 | 0.848 | 0.888 |

**Ruido Salt & Pepper**

| Descriptor | Limpio | Bajo | Medio | Alto |
|---|---|---|---|---|
| Hu | 0.645 | 0.652 | 0.634 | 0.606 |
| Zernike | 0.822 | 0.826 | 0.835 | 0.759 |
| Shape Signature | 0.830 | 0.855 | 0.867 | 0.795 |

> Figuras: `results/part1/cm_*.png` (matrices de confusión),
> `results/part1/robustez_gaussian.png`, `results/part1/robustez_salt_pepper.png`
> (curvas de robustez) y `results/part1/pca_zernike_limpio.png`.

### 3. Análisis

- **Superioridad de la ortogonalidad.** Los **Momentos de Hu** son
  consistentemente el descriptor **más débil** (≈ 0.61–0.65) y el más sensible:
  sus invariantes de alto orden son casi redundantes y muy sensibles a la
  irregularidad del trazo. Los **Momentos de Zernike**, por su base ortogonal,
  mantienen ≈ 0.82 de forma estable, confirmando la conclusión de la guía.
- **Shape Signature** obtiene la mejor precisión global porque trabaja sobre el
  contorno remuestreado y su firma espectral es naturalmente invariante a
  rotación, escala y punto de inicio.
- **Degradación con ruido.** Con Salt & Pepper alto se observa la mayor caída
  (Zernike 0.84→0.76, Shape Signature 0.87→0.79): el ruido impulsivo daña la
  segmentación del contorno. El filtro de mediana mitiga el efecto en niveles
  bajos/medios. El ruido Gaussiano afecta menos porque se promedia.
- **PCA.** La proyección de Zernike (limpio) muestra tres clusters separables que
  el ruido tiende a mezclar.

### 4. Conclusión

El experimento reproduce la conclusión teórica: **Zernike > Hu** en robustez por
su ortogonalidad, y la **firma espectral** (Shape Signature) es un descriptor de
contorno muy competitivo e intrínsecamente invariante. La calidad de la
**segmentación previa** es el factor crítico ante ruido alto.
