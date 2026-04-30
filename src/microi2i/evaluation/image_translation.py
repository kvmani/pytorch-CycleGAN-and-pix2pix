"""Image translation and microscopy-oriented metric helpers."""

from __future__ import annotations

from math import log10, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except Exception:  # optional dependency
    structural_similarity = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _psnr(rmse: float) -> float:
    if rmse <= 0:
        return float("inf")
    return 20.0 * log10(255.0 / rmse)


def _gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to luminance-like grayscale."""

    return (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2])


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.sqrt((gx * gx) + (gy * gy))


def _laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - (4.0 * center)
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    a_std = float(np.std(af))
    b_std = float(np.std(bf))
    if a_std <= 1e-12 or b_std <= 1e-12:
        return None
    return float(np.corrcoef(af, bf)[0, 1])


def _histogram_l1(a: np.ndarray, b: np.ndarray, *, bins: int = 32) -> float:
    a_hist, _ = np.histogram(a, bins=bins, range=(0, 255), density=False)
    b_hist, _ = np.histogram(b, bins=bins, range=(0, 255), density=False)
    a_hist = a_hist.astype(np.float64) / max(float(np.sum(a_hist)), 1.0)
    b_hist = b_hist.astype(np.float64) / max(float(np.sum(b_hist)), 1.0)
    return float(0.5 * np.sum(np.abs(a_hist - b_hist)))


def _cnr_proxy(gray: np.ndarray) -> float:
    p95 = float(np.percentile(gray, 95))
    p5 = float(np.percentile(gray, 5))
    return float((p95 - p5) / (float(np.std(gray)) + 1e-8))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def _sample_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(sqrt(float(np.mean(diff * diff))))
    ssim = None
    if structural_similarity is not None and min(pred.shape[:2]) >= 7:
        ssim = float(structural_similarity(pred, target, channel_axis=2, data_range=255.0))

    pred_gray = _gray(pred)
    target_gray = _gray(target)
    pred_grad = _gradient_magnitude(pred_gray)
    target_grad = _gradient_magnitude(target_gray)
    pred_lap = _laplacian(pred_gray)
    target_lap = _laplacian(target_gray)

    pred_high_frequency = float(np.mean(pred_grad))
    target_high_frequency = float(np.mean(target_grad))
    pred_sharpness = float(np.var(pred_lap))
    target_sharpness = float(np.var(target_lap))

    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": _psnr(rmse),
        "ssim": ssim,
        "gradient_correlation": _pearson(pred_grad, target_grad),
        "edge_mae": float(np.mean(np.abs(pred_grad - target_grad))),
        "histogram_l1": _histogram_l1(pred_gray, target_gray),
        "cnr_proxy_delta": abs(_cnr_proxy(pred_gray) - _cnr_proxy(target_gray)),
        "high_frequency_energy_ratio": _safe_ratio(pred_high_frequency, target_high_frequency),
        "laplacian_sharpness_ratio": _safe_ratio(pred_sharpness, target_sharpness),
    }


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    if values:
        return float(np.mean(values))
    if any(row.get(key) == float("inf") for row in rows):
        return float("inf")
    return None


def evaluate_paired_directories(predictions_dir: str | Path, targets_dir: str | Path) -> dict[str, Any]:
    """Evaluate same-named prediction and target images with scientific metrics."""

    pred_root = Path(predictions_dir)
    target_root = Path(targets_dir)
    if not pred_root.exists() or not target_root.exists():
        return {
            "status": "skipped",
            "reason": "predictions_dir or targets_dir does not exist",
            "predictions_dir": str(pred_root),
            "targets_dir": str(target_root),
            "samples": [],
            "aggregate": {},
        }

    target_by_name = {
        path.name: path
        for path in target_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    rows: list[dict[str, Any]] = []
    for pred_path in sorted(pred_root.iterdir()):
        if not pred_path.is_file() or pred_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target_path = target_by_name.get(pred_path.name)
        if target_path is None:
            continue
        pred = _load_image(pred_path)
        target = _load_image(target_path)
        if pred.shape != target.shape:
            target = np.asarray(Image.open(target_path).convert("RGB").resize((pred.shape[1], pred.shape[0])), dtype=np.float32)
        rows.append({"sample": pred_path.name, **_sample_metrics(pred, target)})

    if not rows:
        return {
            "status": "skipped",
            "reason": "no same-named image pairs found",
            "predictions_dir": str(pred_root),
            "targets_dir": str(target_root),
            "samples": [],
            "aggregate": {},
        }

    return {
        "status": "computed",
        "predictions_dir": str(pred_root),
        "targets_dir": str(target_root),
        "samples": rows,
        "aggregate": {
            "sample_count": len(rows),
            "mae_mean": _mean_optional(rows, "mae"),
            "rmse_mean": _mean_optional(rows, "rmse"),
            "psnr_mean": _mean_optional(rows, "psnr"),
            "ssim_mean": _mean_optional(rows, "ssim"),
            "gradient_correlation_mean": _mean_optional(rows, "gradient_correlation"),
            "edge_mae_mean": _mean_optional(rows, "edge_mae"),
            "histogram_l1_mean": _mean_optional(rows, "histogram_l1"),
            "cnr_proxy_delta_mean": _mean_optional(rows, "cnr_proxy_delta"),
            "high_frequency_energy_ratio_mean": _mean_optional(rows, "high_frequency_energy_ratio"),
            "laplacian_sharpness_ratio_mean": _mean_optional(rows, "laplacian_sharpness_ratio"),
        },
    }
