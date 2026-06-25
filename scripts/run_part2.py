"""
Ruta:           scripts/run_part2.py
Responsabilidad:
    Referencia en PYTHON de la Parte 2 (espejo del núcleo C++). Reconocimiento de
    hojas del Flavia Dataset mediante el Descriptor de Fourier de coordenadas
    complejas y clasificación por distancia Euclídea (KNN, K=7).

    Es DATA-AGNÓSTICO: lee la ruta del dataset por argumento. Cuando coloques el
    Flavia (con all.csv + imágenes) en data/flavia/ (o donde indiques), corre sin
    cambios:
        - Filtra las clases asignadas al G3: 14..20.
        - 80% train / 20% test por clase.
        - Genera matriz de confusión, accuracy, ejemplos de hojas confundidas y
          el modelo entrenado (model.csv) para empaquetar en el móvil.

    Uso:
        python -m scripts.run_part2 [ruta_flavia]
    (por defecto usa config.FLAVIA_DIR = data/flavia)

    Ejecutar desde la raíz del proyecto (vision_project/).
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.descriptors.shape_signature.fourier_complex import (euclidean_distance,
                                                             fourier_descriptor)
from src.preprocessing.leaf_segmentation import extract_largest_contour
from src.evaluation.metrics import plot_confusion_matrix
from src.utils import config
from src.utils.io_utils import ensure_dir


def find_image(root: Path, image_id: str):
    """Busca la imagen 'id' en root/ o root/images/."""
    for cand in (root / image_id, root / "images" / image_id):
        if cand.exists():
            return cand
    return None


def knn_predict(query, train_X, train_y, k=config.KNN_K):
    """Vecino más cercano por distancia Euclídea (voto entre k)."""
    dists = [euclidean_distance(query, x) for x in train_X]
    idx = np.argsort(dists)[:k]
    votes = defaultdict(int)
    for i in idx:
        votes[train_y[i]] += 1
    return max(votes.items(), key=lambda kv: kv[1])[0]


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else config.FLAVIA_DIR
    csv_path = root / config.FLAVIA_CSV_NAME
    out_dir = ensure_dir(config.RESULTS_PART2)

    if not csv_path.exists():
        print("=" * 70)
        print("[AVISO] No se encontró el Flavia Dataset.")
        print(f"        Esperado: {csv_path}")
        print("        Coloca all.csv y las imágenes (id.jpg) en esa carpeta y")
        print("        vuelve a ejecutar:  python -m scripts.run_part2")
        print("        Estructura esperada:")
        print("            data/flavia/all.csv        (columnas: count,id,y)")
        print("            data/flavia/1300.jpg ...   (o data/flavia/images/)")
        print("=" * 70)
        return

    # Leer all.csv y filtrar las clases asignadas.
    import csv as _csv
    rows = []
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            try:
                y = int(r["y"])
            except (KeyError, ValueError):
                continue
            if y in config.FLAVIA_CLASSES:
                rows.append((r["id"], y))
    print(f"[INFO] {len(rows)} imágenes de las clases {config.FLAVIA_CLASSES}")

    # Descriptor por hoja, agrupado por clase.
    by_class = defaultdict(list)
    ok = fail = 0
    for image_id, y in rows:
        path = find_image(root, image_id)
        if path is None:
            fail += 1
            continue
        img = cv2.imread(str(path))
        cnt = extract_largest_contour(img) if img is not None else None
        if cnt is None:
            fail += 1
            continue
        by_class[y].append(fourier_descriptor(cnt))
        ok += 1
    print(f"[INFO] descriptores OK={ok} fallidos={fail}")

    # Split 80/20 por clase.
    train_X, train_y, test_X, test_y = [], [], [], []
    for y, descs in by_class.items():
        n_tr = int(len(descs) * config.TRAIN_RATIO)
        for i, d in enumerate(descs):
            if i < n_tr:
                train_X.append(d); train_y.append(y)
            else:
                test_X.append(d); test_y.append(y)
    print(f"[INFO] train={len(train_X)} test={len(test_X)}")
    if not test_X:
        print("[ERROR] No hay muestras de test. ¿Dataset incompleto?")
        return

    # Guardar el modelo (para el móvil): label,d0,d1,...
    model_path = out_dir / "model.csv"
    with open(model_path, "w") as f:
        for d, y in zip(train_X, train_y):
            f.write(str(y) + "," + ",".join(f"{v:.6f}" for v in d) + "\n")
    print(f"[INFO] modelo guardado en {model_path}")

    # Evaluación.
    classes = config.FLAVIA_CLASSES
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_true, y_pred = [], []
    confused = []   # (id_simulado, real, pred) para ejemplos
    for d, y in zip(test_X, test_y):
        p = knn_predict(d, train_X, train_y)
        y_true.append(cls_idx[y]); y_pred.append(cls_idx[p])
        if p != y:
            confused.append((y, p))

    acc = plot_confusion_matrix(
        y_true, y_pred, [str(c) for c in classes],
        "Flavia (clases 14-20) | Fourier complejo + Euclídea (K=7)",
        out_dir / "confusion_matrix_part2.png")

    print(f"\n[OK] Accuracy = {acc:.4f}")
    print(f"[OK] Matriz de confusión -> {out_dir/'confusion_matrix_part2.png'}")
    if confused:
        print(f"[INFO] {len(confused)} hojas confundidas (real -> pred):")
        for real, pred in confused[:10]:
            print(f"        {real} -> {pred}")


if __name__ == "__main__":
    main()
