#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detecteur_image_ia.core import image_to_gray16, save_gray16_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertit une image en PNG niveau de gris 16 bits.")
    parser.add_argument("input", type=Path, help="Image source")
    parser.add_argument("output", type=Path, help="PNG gris 16 bits de sortie")
    args = parser.parse_args()

    gray = image_to_gray16(args.input)
    save_gray16_png(gray, args.output)


if __name__ == "__main__":
    main()
