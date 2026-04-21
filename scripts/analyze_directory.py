#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detecteur_image_ia.core import (
    image_to_gray16,
    plot_vector_distribution,
    save_gray16_png,
    save_source_pixels,
    save_vector_distribution,
    save_vector_field,
    vector_field_from_gray16,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse toutes les images d'un dossier avec des vecteurs de log-ratio.")
    parser.add_argument("input_dir", type=Path, help="Dossier contenant les images a tester")
    parser.add_argument("output_dir", type=Path, help="Dossier de sortie")
    parser.add_argument("--plot-size", type=int, default=1024, help="Taille en pixels des graphiques carres")
    parser.add_argument("--point-alpha", type=int, default=28, help="Opacite des points, de 1 a 255")
    parser.add_argument("--scale", type=int, default=None, help="Amplitude maximale affichee sur chaque axe")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile utilise si --scale est absent")
    parser.add_argument("--smooth-radius", type=int, default=1, help="Rayon de lissage visuel du graphique")
    args = parser.parse_args()

    images = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"Aucune image trouvee dans {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(images, start=1):
        prefix = f"{index:03d}_{_slugify(image_path.stem)}"
        source_path = args.output_dir / f"{prefix}_source_pixels.npz"
        gray_path = args.output_dir / f"{prefix}_gray16.png"
        vectors_path = args.output_dir / f"{prefix}_vectors.npz"
        distribution_path = args.output_dir / f"{prefix}_distribution.npz"
        plot_path = args.output_dir / f"{prefix}_vectors.png"

        save_source_pixels(image_path, source_path)
        gray = image_to_gray16(image_path)
        save_gray16_png(gray, gray_path)
        vx, vy = vector_field_from_gray16(gray)
        save_vector_field(vx, vy, vectors_path)
        save_vector_distribution(vx, vy, distribution_path)
        plot_vector_distribution(
            vx,
            vy,
            plot_path,
            size=args.plot_size,
            point_alpha=args.point_alpha,
            scale=args.scale,
            percentile=args.percentile,
            smooth_radius=args.smooth_radius,
        )

        print(
            f"{image_path.name} -> {source_path.name}, {gray_path.name}, "
            f"{vectors_path.name}, {distribution_path.name}, {plot_path.name}"
        )


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "image"


if __name__ == "__main__":
    main()
