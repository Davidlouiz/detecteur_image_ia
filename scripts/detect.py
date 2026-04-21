#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detecteur_image_ia.detector import classify_features, features_from_image, iter_image_paths, load_model


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "current_detector.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecteur experimental d'images IA base sur les ratios de pixels.")
    parser.add_argument("input", type=Path, help="Image ou dossier d'images a classer")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Modele JSON entraine")
    parser.add_argument("--csv", type=Path, default=None, help="Fichier CSV de sortie")
    parser.add_argument("--copy-to", type=Path, default=None, help="Copie les images originales dans deux dossiers de controle")
    parser.add_argument("--show-features", action="store_true", help="Affiche aussi les features principales")
    args = parser.parse_args()

    model = load_model(args.model)
    images = iter_image_paths(args.input)
    if not images:
        raise SystemExit(f"Aucune image trouvee dans {args.input}")

    if args.copy_to is not None:
        for label_dir in set(model["labels"].values()):
            shutil.rmtree(args.copy_to / label_dir, ignore_errors=True)

    rows = []
    for image_path in images:
        features = features_from_image(image_path)
        result = classify_features(features, model)
        row = {
            "path": str(image_path),
            "prediction": str(result["prediction"]),
            "score_fake": f"{float(result['score_fake']):.4f}",
            "confidence": f"{float(result['confidence']):.4f}",
            "entropy": f"{features['entropy']:.6f}",
            "unique_ratio": f"{features['unique_ratio']:.6f}",
            "central_symmetry": f"{features['central_symmetry']:.6f}",
            "xaxis_symmetry": f"{features['xaxis_symmetry']:.6f}",
            "yaxis_symmetry": f"{features['yaxis_symmetry']:.6f}",
            "trace": f"{features['trace']:.6f}",
            "determinant": f"{features['determinant']:.6f}",
            "anisotropy": f"{features['anisotropy']:.6f}",
            "corr": f"{features['corr']:.6f}",
            "q99_radius": f"{features['q99_radius']:.6f}",
            "zero_vec": f"{features['zero_vec']:.6f}",
            "angle_entropy": f"{features['angle_entropy']:.6f}",
        }
        rows.append(row)
        print(
            f"{row['prediction']:22s} score_fake={row['score_fake']} "
            f"confidence={row['confidence']}  {image_path}"
        )
        if args.show_features:
            print(
                f"  entropy={row['entropy']} unique={row['unique_ratio']} "
                f"central={row['central_symmetry']} xaxis={row['xaxis_symmetry']} "
                f"yaxis={row['yaxis_symmetry']} trace={row['trace']}"
            )
        if args.copy_to is not None:
            _copy_for_review(image_path, args.copy_to / str(result["prediction"]))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _copy_for_review(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    shutil.copy2(source, target)


if __name__ == "__main__":
    main()
