from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .core import RATIO_LOG_SCALE, image_to_gray16, unique_vector_counts, vector_field_from_gray16


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

COVARIANCE_FEATURE_NAMES = [
    "mean_x",
    "mean_y",
    "cov_xx",
    "cov_xy",
    "cov_yy",
    "std_x",
    "std_y",
    "trace",
    "determinant",
    "eigen_min",
    "eigen_max",
    "corr",
    "anisotropy",
    "principal_cos2",
    "principal_sin2",
    "mean_radius",
    "q50_radius",
    "q90_radius",
    "q99_radius",
    "zero_vec",
    "angle_entropy",
]

RATIO_STATS_FEATURE_NAMES = [
    "unique_ratio",
    "entropy",
    "max_count_ratio",
    "top10_ratio",
    "zero_vec",
    "zero_x",
    "zero_y",
    "mean_abs",
    "q90",
    "q99",
    "mean_x",
    "mean_y",
    "std_x",
    "std_y",
    "corr",
    "anisotropy",
    "angle_entropy",
    "central_symmetry",
    "xaxis_symmetry",
    "yaxis_symmetry",
]

FEATURE_NAMES = RATIO_STATS_FEATURE_NAMES

LOGISTIC_L2 = 0.001
LOGISTIC_STEPS = 5000
LOGISTIC_LEARNING_RATE = 0.05


def iter_image_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"{root} is not a supported image file")
        return [root]

    if not root.is_dir():
        raise ValueError(f"{root} does not exist")

    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def features_from_image(path: str | Path) -> dict[str, float]:
    gray = image_to_gray16(path)
    vx, vy = vector_field_from_gray16(gray)
    return features_from_vectors(vx, vy)


def features_from_vectors(vx: np.ndarray, vy: np.ndarray) -> dict[str, float]:
    vx = np.asarray(vx, dtype=np.int32)
    vy = np.asarray(vy, dtype=np.int32)
    if vx.shape != vy.shape:
        raise ValueError("vx and vy must have the same shape")

    vx_flat = vx.ravel()
    vy_flat = vy.ravel()
    total = vx_flat.size
    if total == 0:
        raise ValueError("vector field is empty")

    # Flatten into a N x 2 cloud of log-ratio vectors. We keep covariance
    # features available, then add the exact ratio-distribution features that
    # currently validate better on the sample set.
    x = vx_flat.astype(np.float64) / RATIO_LOG_SCALE
    y = vy_flat.astype(np.float64) / RATIO_LOG_SCALE
    dx = x - x.mean()
    dy = y - y.mean()
    cov_xx = float(np.mean(dx * dx))
    cov_xy = float(np.mean(dx * dy))
    cov_yy = float(np.mean(dy * dy))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigen_min = max(0.0, float(eigenvalues[0]))
    eigen_max = max(0.0, float(eigenvalues[1]))
    trace = cov_xx + cov_yy
    determinant = max(0.0, cov_xx * cov_yy - cov_xy * cov_xy)
    std_x = math.sqrt(cov_xx)
    std_y = math.sqrt(cov_yy)
    radius = np.sqrt(x * x + y * y)
    nonzero = (vx_flat != 0) | (vy_flat != 0)
    principal_angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    covariance_features = {
        "mean_x": float(x.mean()),
        "mean_y": float(y.mean()),
        "cov_xx": cov_xx,
        "cov_xy": cov_xy,
        "cov_yy": cov_yy,
        "std_x": std_x,
        "std_y": std_y,
        "trace": trace,
        "determinant": determinant,
        "eigen_min": eigen_min,
        "eigen_max": eigen_max,
        "corr": _correlation_from_covariance(cov_xy, std_x, std_y),
        "anisotropy": eigen_max / max(eigen_min, 1e-12),
        "principal_cos2": math.cos(2.0 * principal_angle),
        "principal_sin2": math.sin(2.0 * principal_angle),
        "mean_radius": float(radius.mean()),
        "q50_radius": float(np.quantile(radius, 0.50)),
        "q90_radius": float(np.quantile(radius, 0.90)),
        "q99_radius": float(np.quantile(radius, 0.99)),
        "zero_vec": float(np.mean((vx_flat == 0) & (vy_flat == 0))),
        "angle_entropy": _angle_entropy(vx_flat[nonzero], vy_flat[nonzero]),
    }
    unique_vx, unique_vy, counts = unique_vector_counts(vx_flat, vy_flat)
    probabilities = counts.astype(np.float64) / float(total)
    magnitude = np.maximum(np.abs(x), np.abs(y))

    return {
        **covariance_features,
        "unique_ratio": float(counts.size / total),
        "entropy": _normalized_entropy(probabilities, total),
        "max_count_ratio": float(counts.max() / total),
        "top10_ratio": float(np.sort(counts)[-10:].sum() / total),
        "zero_x": float(np.mean(vx_flat == 0)),
        "zero_y": float(np.mean(vy_flat == 0)),
        "mean_abs": float(np.mean(magnitude)),
        "q90": float(np.quantile(magnitude, 0.90)),
        "q99": float(np.quantile(magnitude, 0.99)),
        "central_symmetry": _symmetry_ratio(unique_vx, unique_vy, counts, mode="central"),
        "xaxis_symmetry": _symmetry_ratio(unique_vx, unique_vy, counts, mode="xaxis"),
        "yaxis_symmetry": _symmetry_ratio(unique_vx, unique_vy, counts, mode="yaxis"),
    }


def train_centroid_model(samples: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")

    labels = {label for label, _ in samples}
    if labels != {"fake", "real"}:
        raise ValueError("training samples must contain exactly 'fake' and 'real' labels")

    matrix = np.array([[features[name] for name in FEATURE_NAMES] for _, features in samples], dtype=np.float64)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (matrix - center) / scale

    centroids: dict[str, list[float]] = {}
    for label in ["fake", "real"]:
        label_rows = z[[sample_label == label for sample_label, _ in samples]]
        centroids[label] = label_rows.mean(axis=0).tolist()

    return {
        "version": 1,
        "kind": "centroid",
        "feature_names": FEATURE_NAMES,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "centroids": centroids,
        "labels": {"fake": "image_ia_probable", "real": "image_reelle_probable"},
        "training_counts": {
            "fake": sum(label == "fake" for label, _ in samples),
            "real": sum(label == "real" for label, _ in samples),
        },
        "note": "Modele experimental base sur les statistiques exactes des vecteurs de log-ratio, pas sur les noms ou metadonnees.",
    }


def train_logistic_model(samples: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")

    labels = {label for label, _ in samples}
    if labels != {"fake", "real"}:
        raise ValueError("training samples must contain exactly 'fake' and 'real' labels")

    matrix = np.array([[features[name] for name in FEATURE_NAMES] for _, features in samples], dtype=np.float64)
    y = np.array([1.0 if label == "fake" else 0.0 for label, _ in samples], dtype=np.float64)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (matrix - center) / scale
    weights, bias = _fit_weighted_logistic_regression(z, y)

    return {
        "version": 2,
        "kind": "logistic",
        "feature_names": FEATURE_NAMES,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "labels": {"fake": "image_ia_probable", "real": "image_reelle_probable"},
        "training_counts": {
            "fake": sum(label == "fake" for label, _ in samples),
            "real": sum(label == "real" for label, _ in samples),
        },
        "regularization_l2": LOGISTIC_L2,
        "note": "Modele experimental base sur les statistiques exactes des vecteurs de log-ratio, pas sur les noms ou metadonnees.",
    }


def save_model(model: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify_features(features: dict[str, float], model: dict[str, Any]) -> dict[str, float | str]:
    names = model["feature_names"]
    vector = np.array([features[name] for name in names], dtype=np.float64)
    center = np.array(model["center"], dtype=np.float64)
    scale = np.array(model["scale"], dtype=np.float64)
    z = (vector - center) / scale

    if model.get("kind") == "logistic":
        weights = np.array(model["weights"], dtype=np.float64)
        score_fake = _sigmoid(float(z @ weights + model["bias"]))
        confidence = abs(score_fake - 0.5) * 2.0
        predicted = "fake" if score_fake >= 0.5 else "real"

        return {
            "prediction": model["labels"][predicted],
            "raw_label": predicted,
            "score_fake": float(score_fake),
            "confidence": float(confidence),
            "distance_fake": float(1.0 - score_fake),
            "distance_real": float(score_fake),
        }

    fake_centroid = np.array(model["centroids"]["fake"], dtype=np.float64)
    real_centroid = np.array(model["centroids"]["real"], dtype=np.float64)
    fake_distance = float(np.sqrt(np.mean((z - fake_centroid) ** 2)))
    real_distance = float(np.sqrt(np.mean((z - real_centroid) ** 2)))
    score_fake = real_distance / (fake_distance + real_distance + 1e-12)
    confidence = abs(score_fake - 0.5) * 2.0
    predicted = "fake" if score_fake >= 0.5 else "real"

    return {
        "prediction": model["labels"][predicted],
        "raw_label": predicted,
        "score_fake": float(score_fake),
        "confidence": float(confidence),
        "distance_fake": fake_distance,
        "distance_real": real_distance,
    }


def leave_one_out_accuracy(samples: list[tuple[str, dict[str, float]]]) -> float:
    if len(samples) < 3:
        return 0.0

    correct = 0
    for index, (expected_label, features) in enumerate(samples):
        training = [sample for sample_index, sample in enumerate(samples) if sample_index != index]
        model = train_logistic_model(training)
        result = classify_features(features, model)
        correct += result["raw_label"] == expected_label
    return correct / len(samples)


def _fit_weighted_logistic_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    sample_count, feature_count = x.shape
    positive_count = y.sum()
    negative_count = sample_count - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("logistic regression needs both classes")

    class_weights = np.where(y == 1.0, sample_count / (2.0 * positive_count), sample_count / (2.0 * negative_count))
    weights = np.zeros(feature_count, dtype=np.float64)
    bias = 0.0

    for _ in range(LOGISTIC_STEPS):
        probabilities = _sigmoid_array(x @ weights + bias)
        error = (probabilities - y) * class_weights
        gradient_weights = (x.T @ error) / sample_count + LOGISTIC_L2 * weights
        gradient_bias = float(error.mean())
        weights -= LOGISTIC_LEARNING_RATE * gradient_weights
        bias -= LOGISTIC_LEARNING_RATE * gradient_bias

    return weights, bias


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _normalized_entropy(probabilities: np.ndarray, base_size: int) -> float:
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0 or base_size <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(base_size))


def _angle_entropy(vx: np.ndarray, vy: np.ndarray) -> float:
    if vx.size == 0:
        return 0.0
    angles = np.arctan2(vy.astype(np.float64), vx.astype(np.float64))
    counts, _ = np.histogram(angles, bins=36, range=(-math.pi, math.pi))
    probabilities = counts[counts > 0].astype(np.float64) / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(36))


def _correlation_from_covariance(cov_xy: float, std_x: float, std_y: float) -> float:
    denominator = std_x * std_y
    if denominator < 1e-12:
        return 0.0
    return float(cov_xy / denominator)


def _symmetry_ratio(vx: np.ndarray, vy: np.ndarray, counts: np.ndarray, *, mode: str) -> float:
    if mode == "central":
        mirrored_vx = -vx
        mirrored_vy = -vy
    elif mode == "xaxis":
        mirrored_vx = vx
        mirrored_vy = -vy
    elif mode == "yaxis":
        mirrored_vx = -vx
        mirrored_vy = vy
    else:
        raise ValueError(f"unknown symmetry mode: {mode}")

    pairs = _structured_pairs(vx, vy)
    mirrored = _structured_pairs(mirrored_vx, mirrored_vy)
    order = np.argsort(pairs)
    pairs_sorted = pairs[order]
    counts_sorted = counts[order]
    positions = np.searchsorted(pairs_sorted, mirrored)
    valid = positions < pairs_sorted.size
    valid[valid] = pairs_sorted[positions[valid]] == mirrored[valid]
    mirrored_counts = np.zeros_like(counts)
    mirrored_counts[valid] = counts_sorted[positions[valid]]
    return float(np.minimum(counts, mirrored_counts).sum() / counts.sum())


def _structured_pairs(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    pairs = np.empty(vx.shape[0], dtype=[("vx", np.int32), ("vy", np.int32)])
    pairs["vx"] = vx.astype(np.int32)
    pairs["vy"] = vy.astype(np.int32)
    return pairs
