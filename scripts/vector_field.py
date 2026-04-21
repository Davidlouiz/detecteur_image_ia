#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detecteur_image_ia.core import load_gray16_png, save_vector_field, vector_field_from_gray16


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcule vx/vy en log-ratio depuis une image gris 16 bits.")
    parser.add_argument("input", type=Path, help="PNG gris 16 bits")
    parser.add_argument("output", type=Path, help="Archive .npz de sortie")
    args = parser.parse_args()

    gray = load_gray16_png(args.input)
    vx, vy = vector_field_from_gray16(gray)
    save_vector_field(vx, vy, args.output)


if __name__ == "__main__":
    main()
