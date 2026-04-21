#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detecteur_image_ia.detector import (
    IMAGE_EXTENSIONS,
    features_from_image,
    leave_one_out_accuracy,
    save_model,
    train_logistic_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraine le detecteur experimental sur des images labellisees.")
    parser.add_argument("output_model", type=Path, help="Modele JSON a ecrire")
    parser.add_argument("--fake-dir", action="append", type=Path, default=[], help="Dossier contenant des images IA")
    parser.add_argument("--real-dir", action="append", type=Path, default=[], help="Dossier contenant des images reelles")
    parser.add_argument("--fake-pattern", action="append", default=[], help="Motif glob pour images IA")
    parser.add_argument("--real-pattern", action="append", default=[], help="Motif glob pour images reelles")
    args = parser.parse_args()

    fake_images = _collect_images(args.fake_dir, args.fake_pattern)
    real_images = _collect_images(args.real_dir, args.real_pattern)
    if not fake_images or not real_images:
        raise SystemExit("Il faut au moins une image fake et une image reelle pour entrainer.")

    samples = []
    for label, paths in [("fake", fake_images), ("real", real_images)]:
        for index, image_path in enumerate(paths, start=1):
            print(f"[{label} {index}/{len(paths)}] {image_path}", file=sys.stderr)
            samples.append((label, features_from_image(image_path)))

    model = train_logistic_model(samples)
    model["training_accuracy_leave_one_out"] = leave_one_out_accuracy(samples)
    save_model(model, args.output_model)

    print(args.output_model)
    print(f"fake={len(fake_images)} real={len(real_images)}")
    print(f"leave_one_out_accuracy={model['training_accuracy_leave_one_out']:.3f}")


def _collect_images(directories: list[Path], patterns: list[str]) -> list[Path]:
    images: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            raise SystemExit(f"Dossier introuvable: {directory}")
        images.update(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    for pattern in patterns:
        images.update(Path(path) for path in glob.glob(pattern) if Path(path).suffix.lower() in IMAGE_EXTENSIONS)

    return sorted(images)


if __name__ == "__main__":
    main()
