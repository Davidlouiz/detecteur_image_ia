#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detecteur_image_ia.core import load_vector_field, plot_vector_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Dessine la distribution des vecteurs vx/vy de log-ratio.")
    parser.add_argument("input", type=Path, help="Archive .npz produite par vector_field.py")
    parser.add_argument("output", type=Path, help="PNG du graphique de sortie")
    parser.add_argument("--size", type=int, default=1024, help="Taille en pixels du graphique carre")
    parser.add_argument("--point-alpha", type=int, default=28, help="Opacite des points, de 1 a 255")
    parser.add_argument("--scale", type=int, default=None, help="Amplitude maximale affichee sur chaque axe")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile utilise si --scale est absent")
    parser.add_argument("--smooth-radius", type=int, default=1, help="Rayon de lissage visuel du graphique")
    args = parser.parse_args()

    vx, vy = load_vector_field(args.input)
    plot_vector_distribution(
        vx,
        vy,
        args.output,
        size=args.size,
        point_alpha=args.point_alpha,
        scale=args.scale,
        percentile=args.percentile,
        smooth_radius=args.smooth_radius,
    )


if __name__ == "__main__":
    main()
