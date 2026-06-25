"""
Ruta:           scripts/run_part1.py
Responsabilidad:
    Orquestar el ANÁLISIS COMPARATIVO DE ROBUSTEZ de la Parte 1:
        Hu Moments vs Zernike Moments vs Shape Signature
    frente a ruido Gaussiano y Salt & Pepper (niveles Bajo/Medio/Alto) y
    rotaciones aleatorias, usando el dataset UPS-Writing-Skills.

    Para cada (tipo de ruido, nivel) y cada descriptor:
        - Construye el conjunto aumentado y extrae características.
        - Entrena un KNN (K=7) y evalúa sobre un test estratificado.
        - Guarda la matriz de confusión y el accuracy en results/part1/.
    Además genera curvas de robustez (accuracy vs nivel) y una proyección PCA.

    Uso:
        python -m scripts.run_part1            (rápido: subconjunto)
        python -m scripts.run_part1 --full     (dataset completo)

    Ejecutar desde la raíz del proyecto (vision_project/).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Permitir 'import src...' al ejecutar como script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation.pipeline import build_feature_set
from src.classifiers.classifier import build_classifier
from src.descriptors.features import DESCRIPTOR_NAMES
from src.evaluation.metrics import (plot_confusion_matrix, plot_pca,
                                    plot_robustness_curve)
from src.utils import config
from src.utils.io_utils import ensure_dir, load_ups_dataset

LEVELS = ["limpio", "bajo", "medio", "alto"]
NOISE_TYPES = ["gaussian", "salt_pepper"]
DESC_LABEL = {"hu": "Hu Moments", "zernike": "Zernike Moments",
              "shape_signature": "Shape Signature"}


def evaluate_descriptor(X, y, groups, class_names, title, out_png):
    """
    Split por GRUPOS (las variantes de una misma imagen base no se reparten entre
    train y test -> sin data leakage) + KNN(K=7) + matriz de confusión.
    Devuelve el accuracy.
    """
    if len(np.unique(y)) < len(class_names) or len(y) < 2 * len(class_names):
        return float("nan")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3,
                            random_state=config.RANDOM_SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    clf = build_classifier("knn")
    clf.fit(X[tr_idx], y[tr_idx])
    y_pred = clf.predict(X[te_idx])
    return plot_confusion_matrix(y[te_idx], y_pred, class_names, title, out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Usar todas las imágenes (más lento).")
    ap.add_argument("--per-class", type=int, default=25,
                    help="Imágenes por clase en modo rápido.")
    args = ap.parse_args()

    out_dir = ensure_dir(config.RESULTS_PART1)
    images, labels, class_names = load_ups_dataset()
    images = np.array(images, dtype=object)
    labels = np.array(labels)

    # Modo rápido: submuestrear por clase para iterar rápido.
    if not args.full:
        keep = []
        for c in range(len(class_names)):
            idx = np.where(labels == c)[0][:args.per_class]
            keep.extend(idx.tolist())
        images, labels = images[keep], labels[keep]

    print(f"[INFO] {len(images)} imágenes base | clases: {class_names}")
    summary = {}

    for noise_type in NOISE_TYPES:
        # robustez[desc][nivel] = accuracy
        robustness = {d: {} for d in DESCRIPTOR_NAMES}

        for level in LEVELS:
            print(f"[RUN] ruido={noise_type:11s} nivel={level}")
            X_by, y, groups = build_feature_set(list(images), list(labels),
                                                noise_type, level)
            for desc in DESCRIPTOR_NAMES:
                title = f"{DESC_LABEL[desc]} | {noise_type} | {level}"
                png = out_dir / f"cm_{noise_type}_{level}_{desc}.png"
                acc = evaluate_descriptor(X_by[desc], y, groups, class_names,
                                          title, png)
                robustness[desc][level] = acc
                print(f"       {desc:16s} acc={acc:.3f}")

            # PCA del espacio Zernike a nivel limpio (clusters base).
            if level == "limpio" and noise_type == "gaussian":
                plot_pca(X_by["zernike"], y, class_names,
                         "PCA - Zernike (limpio)",
                         out_dir / "pca_zernike_limpio.png")

        # Curva de robustez para este tipo de ruido.
        plot_robustness_curve(
            {DESC_LABEL[d]: robustness[d] for d in DESCRIPTOR_NAMES},
            LEVELS, f"Robustez vs ruido {noise_type}",
            out_dir / f"robustez_{noise_type}.png")
        summary[noise_type] = robustness

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Resultados guardados en {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
