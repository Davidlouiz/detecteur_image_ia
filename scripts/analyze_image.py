#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convertit une image en gris 16 bits, calcule les vecteurs de log-ratio, puis genere leur distribution."
    )
    parser.add_argument("input", type=Path, help="Image source")
    parser.add_argument("output_prefix", type=Path, help="Prefixe des fichiers de sortie")
    parser.add_argument("--plot-size", type=int, default=1024, help="Taille en pixels du graphique carre")
    parser.add_argument("--point-alpha", type=int, default=28, help="Opacite des points du graphique, de 1 a 255")
    parser.add_argument("--scale", type=int, default=None, help="Amplitude maximale affichee sur chaque axe")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile utilise si --scale est absent")
    parser.add_argument("--smooth-radius", type=int, default=1, help="Rayon de lissage visuel du graphique")
    args = parser.parse_args()

    source_path = args.output_prefix.with_name(args.output_prefix.name + "_source_pixels.npz")
    gray_path = args.output_prefix.with_name(args.output_prefix.name + "_gray16.png")
    vectors_path = args.output_prefix.with_name(args.output_prefix.name + "_vectors.npz")
    distribution_path = args.output_prefix.with_name(args.output_prefix.name + "_distribution.npz")
    plot_path = args.output_prefix.with_name(args.output_prefix.name + "_vectors.png")

    save_source_pixels(args.input, source_path)

    gray = image_to_gray16(args.input)
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

    print(source_path)
    print(gray_path)
    print(vectors_path)
    print(distribution_path)
    print(plot_path)


if __name__ == "__main__":
    main()
