from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


PathLike = str | Path

REC709_R = 2126
REC709_G = 7152
REC709_B = 722
REC709_SCALE = 10_000
SOURCE_8BIT_MAX = 255
GRAY16_MAX = 65_535
RATIO_LOG_SCALE = 1_000_000
RATIO_LOG_MAX = 16 * RATIO_LOG_SCALE


def image_to_gray16(path: PathLike) -> np.ndarray:
    """Load an image and return one uint16 grayscale value per pixel."""
    with Image.open(path) as image:
        if image.mode in {"I;16", "I;16L", "I;16B"}:
            return np.asarray(image, dtype=np.uint16)

        if image.mode == "I":
            values = np.asarray(image, dtype=np.int64)
            return np.clip(values, 0, 65535).astype(np.uint16)

        if image.mode == "L":
            return (np.asarray(image, dtype=np.uint16) * 257).astype(np.uint16)

        if image.mode in {"LA", "RGBA"}:
            image = _flatten_alpha_on_white(image)

        rgb = image.convert("RGB")
        data = np.asarray(rgb, dtype=np.uint64)
        # Rec. 709 luma:
        # gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
        # The same formula is evaluated with integer weights, then scaled
        # directly to 16 bits without an intermediate 8-bit grayscale image.
        weighted = REC709_R * data[..., 0] + REC709_G * data[..., 1] + REC709_B * data[..., 2]
        denominator = REC709_SCALE * SOURCE_8BIT_MAX
        gray16 = (weighted * GRAY16_MAX + denominator // 2) // denominator
        return gray16.astype(np.uint16)


def save_source_pixels(path: PathLike, output_path: PathLike) -> None:
    """Store the decoded source pixels exactly as Pillow exposes them."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        pixels = np.asarray(image)
        np.savez_compressed(
            output,
            pixels=pixels,
            mode=np.array(image.mode),
            width=np.array(image.width, dtype=np.int32),
            height=np.array(image.height, dtype=np.int32),
            format=np.array(image.format or ""),
        )


def _flatten_alpha_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def save_gray16_png(gray: np.ndarray, path: PathLike) -> None:
    gray = _require_gray16(gray)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="I;16").save(output)


def load_gray16_png(path: PathLike) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)

    if values.ndim != 2:
        raise ValueError(f"{path} is not a single-channel grayscale image")

    if values.dtype == np.uint16:
        return values

    return np.clip(values.astype(np.int64), 0, 65535).astype(np.uint16)


def vector_field_from_gray16(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centered fixed-point log-ratio vectors for neighboring pixels.

    vx = log2((g(x + 1, y) + 1) / (g(x, y) + 1))
    vy = log2((g(x, y + 1) + 1) / (g(x, y) + 1))

    Values are multiplied by RATIO_LOG_SCALE and rounded to int32. The exact
    ratio remains recoverable from the lossless gray16 image.
    """
    gray = _require_gray16(gray).astype(np.uint32)

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        raise ValueError("image must be at least 2x2 pixels")

    origin = gray[:-1, :-1] + 1
    right = gray[:-1, 1:] + 1
    bottom = gray[1:, :-1] + 1
    vx = _fixed_log2_ratio(right, origin)
    vy = _fixed_log2_ratio(bottom, origin)
    return vx, vy


def _fixed_log2_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = numerator.astype(np.float64) / denominator.astype(np.float64)
    return np.rint(np.log2(ratio) * RATIO_LOG_SCALE).astype(np.int32)


def save_vector_field(vx: np.ndarray, vy: np.ndarray, path: PathLike) -> None:
    vx, vy = _require_vector_field(vx, vy)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vx=vx, vy=vy, kind=np.array("log2_ratio_fixed"), scale=np.array(RATIO_LOG_SCALE))


def save_vector_distribution(vx: np.ndarray, vy: np.ndarray, path: PathLike) -> None:
    vx, vy = _require_vector_field(vx, vy)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dist_vx, dist_vy, counts = unique_vector_counts(vx.ravel(), vy.ravel())
    np.savez_compressed(output, vx=dist_vx, vy=dist_vy, count=counts.astype(np.int64))


def load_vector_field(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        vx = data["vx"].astype(np.int32)
        vy = data["vy"].astype(np.int32)
    return _require_vector_field(vx, vy)


def plot_vector_distribution(
    vx: np.ndarray,
    vy: np.ndarray,
    path: PathLike,
    *,
    size: int = 1024,
    point_alpha: int = 28,
    scale: int | None = None,
    percentile: float | None = None,
    smooth_radius: int = 1,
) -> None:
    vx, vy = _require_vector_field(vx, vy)

    if size < 16:
        raise ValueError("size must be at least 16")

    point_alpha = int(np.clip(point_alpha, 1, 255))
    if scale is None:
        max_abs = np.maximum(np.abs(vx), np.abs(vy)).ravel()
        if percentile is None:
            scale = int(max(1, max_abs.max()))
        else:
            scale = int(np.ceil(np.percentile(max_abs, percentile)))
    scale = max(1, int(scale))

    radius = max(1, size // 2 - 1)
    center = size // 2

    vx_plot = np.clip(vx.ravel(), -scale, scale)
    vy_plot = np.clip(vy.ravel(), -scale, scale)
    x = np.rint(center + (vx_plot.astype(np.float64) / scale) * radius).astype(np.int32)
    y = np.rint(center - (vy_plot.astype(np.float64) / scale) * radius).astype(np.int32)
    mask = (0 <= x) & (x < size) & (0 <= y) & (y < size)

    histogram = np.zeros((size, size), dtype=np.uint32)
    np.add.at(histogram, (y[mask], x[mask]), 1)
    density_source = _smooth_histogram(histogram, smooth_radius)

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    occupied = density_source > 0
    if np.any(occupied):
        density = np.zeros((size, size), dtype=np.float64)
        density[occupied] = np.log1p(density_source[occupied]) / np.log1p(density_source.max())
        density = np.maximum(density, point_alpha / 255.0)
        point_color = np.array((0, 0, 0), dtype=np.float64)
        canvas[occupied] = np.rint(
            (1.0 - density[occupied, None]) * canvas[occupied] + density[occupied, None] * point_color
        ).astype(np.uint8)

    _draw_axes(canvas, center)
    _draw_label(canvas, f"+/-{scale}", x=8, y=8)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(output)


def _draw_axes(canvas: np.ndarray, center: int) -> None:
    canvas[center, :, :] = (210, 210, 210)
    canvas[:, center, :] = (210, 210, 210)
    canvas[center - 2 : center + 3, center - 2 : center + 3, :] = (80, 80, 80)


def _smooth_histogram(histogram: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return histogram.astype(np.float64)

    smoothed = histogram.astype(np.float64)
    for _ in range(radius):
        padded = np.pad(smoothed, 1, mode="constant")
        smoothed = (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return smoothed


def unique_vector_counts(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vx = np.asarray(vx, dtype=np.int32).ravel()
    vy = np.asarray(vy, dtype=np.int32).ravel()
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape")

    pairs = np.empty(vx.shape[0], dtype=[("vx", np.int32), ("vy", np.int32)])
    pairs["vx"] = vx
    pairs["vy"] = vy
    unique, counts = np.unique(pairs, return_counts=True)
    return unique["vx"].astype(np.int32), unique["vy"].astype(np.int32), counts.astype(np.int64)


_DIGITS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "+": ("000", "010", "111", "010", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "/": ("001", "001", "010", "100", "100"),
}


def _draw_label(canvas: np.ndarray, text: str, *, x: int, y: int, scale: int = 3) -> None:
    cursor = x
    for char in text:
        pattern = _DIGITS.get(char)
        if pattern is None:
            cursor += 2 * scale
            continue
        for row_index, row in enumerate(pattern):
            for column_index, value in enumerate(row):
                if value == "1":
                    y0 = y + row_index * scale
                    x0 = cursor + column_index * scale
                    canvas[y0 : y0 + scale, x0 : x0 + scale, :] = (70, 70, 70)
        cursor += 4 * scale


def _require_gray16(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError("gray image must have exactly one channel")
    if gray.dtype != np.uint16:
        raise ValueError("gray image must use dtype uint16")
    return gray


def _require_vector_field(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape")
    if vx.ndim != 2:
        raise ValueError("vx and vy must be two-dimensional")
    return vx.astype(np.int32), vy.astype(np.int32)
