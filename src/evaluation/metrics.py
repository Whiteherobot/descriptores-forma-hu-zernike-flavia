"""
Ruta:           src/evaluation/metrics.py
Responsabilidad:
    Evaluación y visualización de resultados de la Parte 1:
        - Matriz de confusión (por nivel de ruido).
        - Accuracy (precisión global).
        - Gráfico comparativo de robustez (accuracy vs nivel de ruido).
        - Proyección PCA 2-D para observar cómo el ruido "mezcla" los clusters
          (Recomendación de la guía: análisis estadístico con PCA).

    Unidad: 3 (validación con matrices de confusión y análisis de dispersión).
"""
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")            # backend sin ventana (para guardar PNG en disco)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names: List[str],
                          title: str, out_path: Path) -> float:
    """Dibuja y guarda la matriz de confusión. Devuelve el accuracy."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"{title}\nAccuracy = {acc:.3f}")

    # Anotar cada celda con su conteo.
    thr = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return acc


def plot_robustness_curve(results: Dict[str, Dict[str, float]],
                          levels: List[str], title: str, out_path: Path) -> None:
    """
    results : { descriptor -> { nivel -> accuracy } }
    Dibuja accuracy vs nivel de ruido para comparar la robustez de los
    descriptores (Hu vs Zernike vs Shape Signature).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for desc, by_level in results.items():
        ys = [by_level[l] for l in levels]
        ax.plot(levels, ys, marker="o", label=desc)
    ax.set_xlabel("Nivel de ruido")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pca(X: np.ndarray, y: np.ndarray, class_names: List[str],
             title: str, out_path: Path) -> None:
    """Proyección PCA 2-D del espacio de características."""
    if X.shape[0] < 3:
        return
    p = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, name in enumerate(class_names):
        m = y == label
        ax.scatter(p[m, 0], p[m, 1], s=14, alpha=0.6, label=name)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
